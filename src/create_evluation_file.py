import numpy as numpy
import os


dataset_dir = "/home/kevin/DATA/DNIM"
time_stamp_dir = os.path.join(dataset_dir, "time_stamp")
image_dir = os.path.join(dataset_dir, 'Image')
output_dit = "./assets"

files = [x for x in os.listdir(time_stamp_dir) if x.endswith(".txt")]

for i, file in enumerate(files):

    folder = file.split('.')
    folder = folder[0]

    file_dir = os.path.join(time_stamp_dir, file)
    fo = open(os.path.join(output_dit, str(folder)+'.txt'), 'w')

    with open(file_dir, 'r') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            new_line = line.split(' ')[0]
            print(new_line)
            if(len(new_line)):
                new_line += '\n'
                fo.writelines(os.path.join(image_dir, folder, 'ref.jpg ') + os.path.join(image_dir, folder, new_line))

    fo.close()

