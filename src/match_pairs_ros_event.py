#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

#from pathlib import Path
import argparse
from fileinput import filename
import random
import numpy as np
import matplotlib.cm as cm
import torch
import sys
import rospy
import cv2
import time
from cv2 import resize
from match_utils import (
    load_model_eventpoint,
    get_inp,
    process,
    extract_keypoints_and_descriptors,
    match_descriptors,
    match2array,
    compute_homography,
)


# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from nn_matcher.srv import *   
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nn_matcher.msg import nnfeature
from nn_matcher.msg import nnfeaturearray

from models.matching import Matching
from models.matching1 import Matching1
from models.utils import (make_matching_plot, AverageTimer, image2tensor)

torch.set_grad_enabled(False)

class NN_Matching:

    def __init__(self):

        parser = argparse.ArgumentParser(description='Image pair matching and pose evaluation with SuperGlue', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # SuperPoint & SuperGlue parameters
        parser.add_argument('--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt', help='Path to the list of image pairs')
        parser.add_argument('--resize', type=int, default=[672, 376] , help='Resize the input image before running inference. If -1, do not resize')
        parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor', help='SuperGlue weights')
        parser.add_argument('--max_keypoints', type=int, default=240, help='Maximum number of keypoints detected by Superpoint, -1 keeps all keypoints)')
        parser.add_argument('--keypoint_threshold', type=float, default=0.005, help='SuperPoint keypoint detector confidence threshold')
        parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
        parser.add_argument('--sinkhorn_iterations', type=int, default=20, help='Number of Sinkhorn iterations performed by SuperGlue')
        parser.add_argument('--match_threshold', type=float, default=0.1, help='SuperGlue match threshold')
        parser.add_argument('--viz', type=bool, default=True, help='Use faster image visualization with OpenCV instead of Matplotlib')
        parser.add_argument('--viz_extension', type=str, default='png', choices=['png', 'pdf'], help='Visualization file extension. Use pdf for highest-quality.')
        parser.add_argument('--force_cpu', default=False, help='Force pytorch to run in CPU mode.')
        parser.add_argument('--descriptor_only', type=bool, default=True, help='Superpoint descriptor only + NN matcher.')
        parser.add_argument('--superpoint', choices={'official', 'dark'}, default='dark', help='SuperPoint weights')
        parser.add_argument('--mask', type=float, default=0.65, help='Create a mask to get ride of ground.')
        # V-T&R parameters
        parser.add_argument('--maxVerticalDifference', type=int, default=10)
        parser.add_argument('--numBins', type=int, default=41) # 73 / 41
        parser.add_argument('--granlarity', type=int, default=20)
        parser.add_argument('--panorama', type=bool, default=False, help='use fisheye camera.') # [720, 180]  / [640, 350]

        self.args = parser.parse_args()
        print(self.args)

        self.CUDA = True
        self.HOMO = True
        rospy.init_node('nn_image_matcher', anonymous=True)

        # Load the SuperPoint and SuperGlue models.
        self.device = 'cuda' if torch.cuda.is_available() and not self.args.force_cpu else 'cpu'
        print('Running inference on device \"{}\"'.format(self.device))
        config = {
            'superpoint': {
                'weights': self.args.superpoint,
                'nms_radius': self.args.nms_radius,
                'keypoint_threshold': self.args.keypoint_threshold,
                'max_keypoints': self.args.max_keypoints
            },
            'superglue': {
                'weights': self.args.superglue,
                'sinkhorn_iterations': self.args.sinkhorn_iterations,
                'match_threshold': self.args.match_threshold,
            }
        }
        self.model = load_model_eventpoint('./pretrain/EventPoint.pth.tar', self.CUDA)
        self.matching = Matching(config, destcriptor_only=self.args.descriptor_only).eval().to(self.device)
        self.matching1 = Matching1(config, destcriptor_only=self.args.descriptor_only).eval().to(self.device)
        self.imagemap = []
        self.mapdesc0 = []
        self.mapkpts3d = []
        self.mapkpts0 = []
        self.nn_matching_srv = rospy.Service('NN_Image_Matching', NNImageMatching, self.matching_pair)
        self.nn_select_srv = rospy.Service('NN_Image_Select', NNImageSelect, self.matching_pair_select)
        self.nn_matching_stereo_srv = rospy.Service('NN_Image_Matching_Stereo', NNImageMatchingStereo, self.matching_pair_stereo)
        self.matched_feats_pub = rospy.Publisher('/matched_features', Image, queue_size = 1)
        self.i=0

        print("Ready to localize the robot!")
        rospy.spin()


    def building_histogram(self, kpts0, kpts1):
        # histogram = np.zeros(self.args.numBins, dtype=int)
        kpts0 = np.array(kpts0)
        kpts1 = np.array(kpts1)
        # print(kpts0.shape)
        # print(kpts1.shape)
        differenceX = kpts0[:, 0] - kpts1[:, 0]
        differenceY = kpts0[:, 1] - kpts1[:, 1]
        invaild = abs(differenceY) > self.args.maxVerticalDifference

        if self.args.panorama is True:
            differenceX[np.nonzero(differenceX>0.5*self.args.resize[0])] -= self.args.resize[0]
            differenceX[np.nonzero(differenceX<-0.5*self.args.resize[0])] += self.args.resize[0]

        differences = differenceX
        differences[invaild] = -1000000

        index = (differenceX + self.args.granlarity / 2) / self.args.granlarity + self.args.numBins / 2
        index = index[~invaild]
        # unique_index, counts_index = np.unique(index, return_counts=True)
        span = (self.args.numBins * self.args.granlarity) / 2
        # histogram, bin_edges = np.histogram(index, bins=self.args.numBins, range=(-span, span))

        return differences, []


    def matching_pair(self, req):
        timer = AverageTimer(newline=True)

        bridge = CvBridge()
        #cv_img_map = bridge.imgmsg_to_cv2(req.image_map, "passthrough")
        cv_img_map = self.imagemap[int(req.image_map_index)]
        cv_img_camera = bridge.imgmsg_to_cv2(req.image_camera, "passthrough")
        #cv_img_camera = cv_img_camera1[:,:672]
        #print(desc3.shape)
        #print(type(desc3))
        grayim1 = cv_img_map.astype('uint8')
        grayim2 = cv_img_camera.astype('uint8')
        gray1 = (grayim1.astype('float32') / 255.)
        gray2 = (grayim2.astype('float32') / 255.)
        """ if(self.args.mask < 1.0):
            num_row = int(cv_img_map.shape[0]*self.args.mask)
            
            if self.args.panorama:
                cv_img_map = cv_img_map[-num_row:, :]
                cv_img_camera = cv_img_camera[-num_row:, :]
            else:
                cv_img_map = cv_img_map[:num_row, :]
                cv_img_camera = cv_img_camera[:num_row, :] """

        inp2 = get_inp(gray2, self.CUDA)

        # semi, coarse_desc = self.model.forward(inp1)
        semi_base, coarse_desc_base = self.model.forward(inp2)
        # heatmap, xs, ys = process(semi)
        heatmap_base, xs_base, ys_base = process(semi_base)
        # coarse_desc = coarse_desc.detach().cpu().numpy().squeeze().transpose(1,2,0)
        coarse_desc_base = coarse_desc_base.detach().cpu().numpy().squeeze().transpose(1,2,0)
        # kp1, desc1 = extract_keypoints_and_descriptors(heatmap, coarse_desc)
        kp1 = self.mapkpts0[int(req.image_map_index)] 
        desc1 = self.mapdesc0[int(req.image_map_index)]
        kp2, desc2 = extract_keypoints_and_descriptors(heatmap_base, coarse_desc_base)
        if np.array(kp1).shape == (0,) or np.array(kp2).shape == (0,) :
            res = NNImageMatchingResponse()
        

            feature = nnfeature()
            feature.nnpoints2d = []
            all_nnpoints3d = []
            feature.nnpoints3d=[]
            res.differences = []


            return res
        m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
        if self.HOMO:
            H, inliers = compute_homography(m_kp1, m_kp2)
            matches = np.array(matches)[inliers.astype(bool)].tolist()
            kpc_1, kpc_2 = match2array(matches, kp1, kp2)

        if np.array(matches).shape == (0,):
            differences = []
        else:
            differences, _ = self.building_histogram(kpc_1, kpc_2)
        timer.update('matcher') 

        visualization_image = cv2.drawMatches(grayim1, kp1, grayim2, kp2, matches, None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=0)
        visualization_image = bridge.cv2_to_imgmsg(visualization_image, encoding="passthrough")
        self.matched_feats_pub.publish(visualization_image)

        timer.update('viz_match')

        timer.print('Finished matching pair')
        res = NNImageMatchingResponse()
        
        #featurearray = nnfeaturearray()
        for i in range(len(kpc_2)):
            feature = nnfeature()
            feature.nnpoints2d = []
            all_nnpoints3d = []
            feature.nnpoints3d=[]
            res.differences = differences


        return res
    def matching_pair_select(self, req):
        timer = AverageTimer(newline=True)

        bridge = CvBridge()
        cv_img_stereo = bridge.imgmsg_to_cv2(req.image_camera, "passthrough")
        """ if(self.args.mask < 1.0):
            num_row = int(cv_img_map.shape[0]*self.args.mask)
            
            if self.args.panorama:
                cv_img_map = cv_img_map[-num_row:, :]
                cv_img_camera = cv_img_camera[-num_row:, :]
            else:
                cv_img_map = cv_img_map[:num_row, :]
                cv_img_camera = cv_img_camera[:num_row, :] """
        grayim2 = cv_img_stereo.astype('uint8')
        gray2 = (grayim2.astype('float32') / 255.)
        inp1 = get_inp(gray2, self.CUDA)
        semi, coarse_desc = self.model.forward(inp1)
        heatmap, xs, ys = process(semi)
        coarse_desc = coarse_desc.detach().cpu().numpy().squeeze().transpose(1,2,0)
        kp1, desc1 = extract_keypoints_and_descriptors(heatmap, coarse_desc)
        kp1=np.array(kp1)

        res1 = NNImageSelectResponse()

        
        res1.kpNum=len(kp1)
        print(len(kp1))
        

        return res1
    def matching_pair_stereo(self, req):
        timer = AverageTimer(newline=True)

        bridge = CvBridge()
        cv_img_stereo = bridge.imgmsg_to_cv2(req.image_stereo, "passthrough")
 


        """ if(self.args.mask < 1.0):
            num_row = int(cv_img_map.shape[0]*self.args.mask)
            
            if self.args.panorama:
                cv_img_map = cv_img_map[-num_row:, :]
                cv_img_camera = cv_img_camera[-num_row:, :]
            else:
                cv_img_map = cv_img_map[:num_row, :]
                cv_img_camera = cv_img_camera[:num_row, :] """
        grayim2 = cv_img_stereo.astype('uint8')
        gray2 = (grayim2.astype('float32') / 255.)
        inp1 = get_inp(gray2, self.CUDA)
        semi, coarse_desc = self.model.forward(inp1)

        heatmap, xs, ys = process(semi)

        coarse_desc = coarse_desc.detach().cpu().numpy().squeeze().transpose(1,2,0)

        kp1, desc1 = extract_keypoints_and_descriptors(heatmap, coarse_desc)           
        visualization_image = bridge.cv2_to_imgmsg(cv_img_stereo, encoding="passthrough")
        self.matched_feats_pub.publish(visualization_image)
        res1 = NNImageMatchingStereoResponse()
        feature = nnfeature()
        if req.reloadmap is True:

            self.imagemap = []
            self.mapdesc0 = []
            self.mapkpts3d = []
            self.mapkpts0 = []
        self.imagemap.append(cv_img_stereo)
        self.mapdesc0.append(desc1)
        self.mapkpts0.append(kp1)
        res1.mapnum = len(self.imagemap)
        print('SuperGlue')

        return res1

    def match_descriptors1(self, kp1, desc1, kp2, desc2, keep=0.9):
        # Match the keypoints with the warped_keypoints with nearest neighbor search
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)

        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:int(len(matches)*keep)]

        matches_idx = np.array([m.queryIdx for m in good])
        m_kp1 = [kp1[idx] for idx in matches_idx]
        matches_idx = np.array([m.trainIdx for m in good])
        m_kp2 = [kp2[idx] for idx in matches_idx]
        confidence = 1 - np.array([m.distance for m in good])

        pts1 = np.array(m_kp1, dtype='float32')
        pts2 = np.array(m_kp2, dtype='float32')

        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)
        new_points1, new_points2, newer_matches = [], [], []

        for i in range(len(good)):
            if mask[i] == 1:
                new_points1.append(kp1[good[i].queryIdx])
                new_points2.append(kp2[good[i].trainIdx])
                #new_desc.append(desc1[good[i].queryIdx])
                newer_matches.append(good[i])

        return new_points1, new_points2, confidence, newer_matches

    def match_descriptors_stereo(self, kp1, desc1, kp2, desc2, keep=0.9):
        # Match the keypoints with the warped_keypoints with nearest neighbor search
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)

        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:int(len(matches)*keep)]

        matches_idx = np.array([m.queryIdx for m in good])
        m_kp1 = [kp1[idx] for idx in matches_idx]
        matches_idx = np.array([m.trainIdx for m in good])
        m_kp2 = [kp2[idx] for idx in matches_idx]
        confidence = 1 - np.array([m.distance for m in good])

        return m_kp1, m_kp2, confidence, good

    def caldepth(self,keypoints1,keypoints2):
        K1 = np.array([[264.075,0, 339.275], [0,264.0325,186.5715], [0, 0, 1]], dtype=np.float32)
        K2= np.array([[263.52 , 0,341.0725], [0,263.49 , 186.186], [0, 0, 1]], dtype=np.float32)
        #K1 = np.array([[528.15,0, 647.55], [0,528.065,358.143], [0, 0, 1]], dtype=np.float32)
        #K2= np.array([[527.04 , 0,651.145], [0,526.98 , 357.372], [0, 0, 1]], dtype=np.float32)
        #K1 = np.array([[1056.3,0, 978.1], [0,1056.13,539.286], [0, 0, 1]], dtype=np.float32)
        #K2= np.array([[1054.08 , 0,985.29], [0,1053.96 , 537.744], [0, 0, 1]], dtype=np.float32)       
        T1 = np.array([[1, 0, 0, 0], 
                       [0, 1, 0, 0], 
                       [0, 0, 1, 0]], dtype=np.float32)
        Rot=np.array([0.00401377,-0.00644605,-0.000220865],dtype=np.float32)
        #trans=np.array([[-119.923],[-0.0954872],[-0.54345]],dtype=np.float32)
        R22, _ = cv2.Rodrigues(Rot)
        #T2 = np.hstack((R22, trans))

        T10 = np.array([119.923,0.0954872,0.54345])
        RT1 = np.zeros((3,4))
        RT1[:3,:3] = R22
        RT1[:3, 3] = -T10
        P1 = np.dot(K2, RT1)




        #T2 = np.concatenate((R22, trans), axis=1)
        P0=np.dot(K1,T1)
        #P1=np.dot(K2,T2)
        #campoints1,campoints2=[],[]
        #for i in range(len(keypoints1)):
           # campoints1.append((self.pixel2cam(keypoints1[i],K1)))
           # campoints2.append((self.pixel2cam(keypoints2[i],K2)))
        #pts1 = np.array(campoints1, dtype='float32')  
        #pts2 = np.array(campoints2, dtype='float32')  
        
        
        #for i in range(len(keypoints1)):

  
        X=cv2.triangulatePoints(P0, P1,keypoints1.T,keypoints2.T)
        X /= X[3]
        Y = X.T

        return Y.tolist()


if __name__ == '__main__':
    dr = NN_Matching()
    


