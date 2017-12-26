#! /usr/bin/env python3

import numpy as np
import cv2
import glob
import os
from utils import *
from augment_data import *
from matplotlib import pyplot as plt
import random

def draw_ellipses(img, metadata):
    draw_img = img.copy()
    for md in metadata:
        [cx, cy, a, b, angle, rings] = md
        draw_ellipse(draw_img, [cx, cy], [a,b], angle, color=red, thickness=2)
    return draw_img



def JohnLaRooy_duplicates(L):
    # from https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list
    seen = set()
    seen2 = set()
    seen_add = seen.add
    seen2_add = seen2.add
    for item in L:
        if item in seen:
            seen2_add(item)
        else:
            seen_add(item)
    return list(seen2)


if __name__ == "__main__":
    random.seed(1)
    path = 'Train/'
    img_file_list = sorted(glob.glob(path+'steelpan*.png'))
    txt_file_list = sorted(glob.glob(path+'steelpan*.txt'))
    assert( len(img_file_list) == len(txt_file_list))

    for i in [1]:#range(len(img_file_list)):
        img_filename = img_file_list[i]
        txt_filename = txt_file_list[i]

        prefix = os.path.splitext(img_filename)[0]
        print(" img_filename = ",img_filename,", prefix = ",prefix)
        img =  cv2.imread(img_filename)
        metadata = read_metadata(txt_filename)
        cv2.imshow(prefix,draw_ellipses(img, metadata))

        img_count = 0
        names = []
        # flip  [ vertical, horizontal, both v & h ]
        for flip_param in [-2,0,1,-1]:
            flip_img, flip_metadata, flip_prefix = flip_image(img, metadata, prefix, flip_param)
            blur_img = flip_img.copy()
            blur_prefix = flip_prefix
            blur_metadata = list(flip_metadata)
            for do_blur in [False, True]:
                if do_blur:
                    blur_img, blur_metadata, blur_prefix = blur_image( blur_img, flip_metadata, flip_prefix)
                for irot in range(5):
                    if (0 == irot):
                        rot_angle = 0
                    else:
                        rot_angle = np.sign(random.random()-0.5)*(1 + 5*np.random.random())  # rotate by 1 to 6 degrees, either direction
                    rot_img, rot_metadata, rot_prefix = rotate_image( blur_img, blur_metadata, blur_prefix, rot_angle)
                    inv_img = rot_img.copy()
                    inv_metadata = list(rot_metadata)
                    inv_prefix = rot_prefix
                    for do_inv in [False, True]:
                        if (do_inv):
                            inv_img, inv_metadata, inv_prefix = invert_image( rot_img, rot_metadata, rot_prefix)
                        names.append(inv_prefix)
                        cv2.imshow(inv_prefix, draw_ellipses(inv_img, inv_metadata))
                        img_count += 1
                        print(img_count, flip_param, do_blur, irot, do_inv, inv_prefix)

        unique = len(set(names))
        dupes = JohnLaRooy_duplicates(names)
        print(" generated ",img_count," new images (len = ",len(names),") and ",unique," unique names.  duplicates are ",dupes)

    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
