#! /usr/bin/env python3

# Implements data-augmentation in the form of flipping images vertically, horizontally, and "both"
# when images are altered, metadata (in accompanying text files) also needs to be altered

import numpy as np
import cv2
import glob
import os
from utils import *


def read_metadata(txt_filename):
    f = open(txt_filename, "r")
    lines = f.readlines()
    f.close()

    xmin = 99999
    arrs = []
    for j in range(len(lines)):
        line = lines[j]   # grab a line
        line = line.translate({ord(c): None for c in '[] '})     # strip unwanted chars in unicode line
        string_vars = line.split(sep=',')
        vals = [int(numeric_string) for numeric_string in string_vars]
        subarr = vals[0:vars_per_pred]  # note vars_per_pred includes a slot for no-object, but numpy slicing convention means subarr will be vars_per_pred-1 elements long
        [cx, cy, a, b, angle, num_rings] = subarr
        #tmp_arr = [cx, cy, a, b, np.cos(2*np.deg2rad(angle)), np.sin(2*np.deg2rad(angle)), 0, num_rings]
        # Input format (from file) is [cx, cy,  a, b, angle, num_rings]
        arrs.append(subarr)
    return arrs



def augment(path='Train/'):
    img_file_list = sorted(glob.glob(path+'steelpan*.bmp'))
    txt_file_list = sorted(glob.glob(path+'steelpan*.txt'))
    assert( len(img_file_list) == len(txt_file_list))

    for i in range(len(img_file_list)):
        img_filename = img_file_list[i]
        txt_filename = txt_file_list[i]

        prefix = os.path.splitext(img_filename)[0]
        print(" img_filename = ",img_filename,", prefix = ",prefix)
        img =  cv2.imread(img_filename)
        height, width, channels = img.shape
        #print(" height, width, channels = ",height, width, channels)
        metadata = read_metadata(txt_filename)
        print(" metadata = ",metadata)

        # flip  [ vertical, horizontal, both v & h ]
        for flip_param in [0,1,-1]:
            img_flip = img.copy()
            flip_metadata = list(metadata)  # copy
            img_flip = cv2.flip( img_flip, flip_param )
            caption = ""
            for an in range(len(flip_metadata)):
                # parse_txt_file gives us md =  [cx, cy, a, b, cos(2*angle), sin(2*angle), 0 (noobj=0, i.e. existence), num_rings]
                [cx, cy, a, b, angle, rings] =  flip_metadata[an]
                cos2t, sin2t = np.cos(2*np.deg2rad(angle)), np.sin(2*np.deg2rad(angle))

                if (flip_param in [0,-1]):
                    cy = height - cy
                    sin2t *= -1       # flip the sin
                if (flip_param in [1,-1]):
                    cx = width - cx
                    cos2t *= -1       # flip the cos

                angle = int(np.rad2deg( np.arctan2(sin2t,cos2t)/2.0 ))
                if (angle < 0):
                    angle += 180

                # output metadata format is md = [cx, cy,  a, b, angle, num_rings]
                this_caption = "[{0}, {1}, {2}, {3}, {4}, {5}]".format(cx, cy, a, b, angle, rings)
                if (an > 0):
                    caption+="\n"
                caption += this_caption

            if (0==flip_param):
                new_prefix = prefix + "_v"
            elif (1==flip_param):
                new_prefix = prefix + "_h"
            else:
                new_prefix = prefix + "_b"
            print("      new_prefix = ",new_prefix)
#            print("      caption = ",caption)
            with open(new_prefix+".txt", "w") as text_file:
                text_file.write(caption)
            cv2.imwrite(new_prefix+'.bmp',img_flip)


if __name__ == "__main__":
    augment()
