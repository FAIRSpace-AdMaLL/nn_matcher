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

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import sys
import rospy
import cv2

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from nn_matcher.srv import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from models.matching import Matching
from models.utils import (make_matching_plot, AverageTimer, image2tensor)

torch.set_grad_enabled(False)

class NN_Matching:

    def __init__(self):

        parser = argparse.ArgumentParser(description='Image pair matching and pose evaluation with SuperGlue', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # SuperPoint & SuperGlue parameters
        parser.add_argument('--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt', help='Path to the list of image pairs')
        parser.add_argument('--resize', type=int, default=[468, 288], help='Resize the input image before running inference. If -1, do not resize')
        parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor', help='SuperGlue weights')
        parser.add_argument('--max_keypoints', type=int, default=500, help='Maximum number of keypoints detected by Superpoint, -1 keeps all keypoints)')
        parser.add_argument('--keypoint_threshold', type=float, default=0.005, help='SuperPoint keypoint detector confidence threshold')
        parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
        parser.add_argument('--sinkhorn_iterations', type=int, default=20, help='Number of Sinkhorn iterations performed by SuperGlue')
        parser.add_argument('--match_threshold', type=float, default=0.1, help='SuperGlue match threshold')
        parser.add_argument('--viz', type=bool, default=True, help='Use faster image visualization with OpenCV instead of Matplotlib')
        parser.add_argument('--viz_extension', type=str, default='png', choices=['png', 'pdf'], help='Visualization file extension. Use pdf for highest-quality.')
        parser.add_argument('--force_cpu', default=True, help='Force pytorch to run in CPU mode.')
        # V-T&R parameters
        parser.add_argument('--maxVerticalDifference', type=int, default=10)
        parser.add_argument('--numBins', type=int, default=41)
        parser.add_argument('--granlarity', type=int, default=20)

        self.args = parser.parse_args()
        print(self.args)

        rospy.init_node('nn_image_matcher', anonymous=True)

        # Load the SuperPoint and SuperGlue models.
        self.device = 'cuda' if torch.cuda.is_available() and not self.args.force_cpu else 'cpu'
        print('Running inference on device \"{}\"'.format(self.device))
        config = {
            'superpoint': {
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
        self.matching = Matching(config).eval().to(self.device)

        self.nn_matching_srv = rospy.Service('NN_Image_Matching', NNImageMatching, self.matching_pair)
        self.matched_feats_pub = rospy.Publisher('/matched_features', Image, queue_size = 1)

        print("Ready to localize the robot!")
        rospy.spin()


    def building_histogram(self, kpts0, kpts1):
        # histogram = np.zeros(self.args.numBins, dtype=int)
        differenceX = kpts0[:, 0] - kpts1[:, 0]
        differenceY = kpts0[:, 1] - kpts1[:, 1]
        invaild = abs(differenceY) > self.args.maxVerticalDifference

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
        cv_img_map = bridge.imgmsg_to_cv2(req.image_map, "passthrough")
        cv_img_camera = bridge.imgmsg_to_cv2(req.image_camera, "passthrough")

        image0, inp0, scales0 = image2tensor(cv_img_map, self.device, self.args.resize, False)
        image1, inp1, scales1 = image2tensor(cv_img_camera, self.device, self.args.resize, False)

        # Perform the matching.
        pred = self.matching({'image0': inp0, 'image1': inp1})

        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        differences, _ = self.building_histogram(kpts0, kpts1)
        print(differences)

        if self.args.viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]

            # Display extra parameter info.
            k_thresh = self.matching.superpoint.config['keypoint_threshold']
            m_thresh = self.matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
            ]

            is_saving = True
            visualization_image = make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, './current.png', True, True, False, 'Matches', small_text, is_saving)

            visualization_image = bridge.cv2_to_imgmsg(visualization_image, encoding="passthrough")
            self.matched_feats_pub.publish(visualization_image)

            timer.update('viz_match')

            timer.print('Finished matching pair')


        res = NNImageMatchingResponse()
        res.differences = differences.tolist()
        res.histogram = []

        return res

if __name__ == '__main__':
    dr = NN_Matching()
