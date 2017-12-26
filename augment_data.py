#! /usr/bin/env python3

# Implements data-augmentation in the form of flipping images vertically, horizontally, and "both"
# when images are altered, metadata (in accompanying text files) also needs to be altered
# By default, only affects *Training* data.  Leaves Val & Test alone

import numpy as np
import cv2
import glob
import os
from utils import *
import random


def read_metadata(txt_filename):
    f = open(txt_filename, "r")
    lines = f.readlines()
    f.close()
    #print("    Reading from text file ",txt_filename)
    xmin = 99999
    arrs = []
    for j in range(len(lines)):
        line = lines[j]   # grab a line
        line = line.translate({ord(c): None for c in '[] '})     # strip unwanted chars in unicode line
        string_vars = line.split(sep=',')
        #print("string_vars = ",string_vars)
        vals = [int(round(float(numeric_string))) for numeric_string in string_vars]
        subarr = vals[0:vars_per_pred]  # note vars_per_pred includes a slot for no-object, but numpy slicing convention means subarr will be vars_per_pred-1 elements long
        [cx, cy, a, b, angle, num_rings] = subarr
        #tmp_arr = [cx, cy, a, b, np.cos(2*np.deg2rad(angle)), np.sin(2*np.deg2rad(angle)), 0, num_rings]
        # Input format (from file) is [cx, cy,  a, b, angle, num_rings]
        arrs.append(subarr)
    return arrs


def caption_from_metadata(metadata):  # converts metadata list-of-lists to caption string which is one antinode per line
    caption = ""
    for an in range(len(metadata)):
        [cx, cy, a, b, angle, rings] = metadata[an]
        this_caption = "[{0}, {1}, {2}, {3}, {4}, {5}]".format(cx, cy, a, b, angle, rings)
        if (an > 0):
            caption +="\n"
        caption += this_caption
    return caption


def cleanup_angle(angle):
    while (angle < 0):
        angle += 180
    while (angle >= 180):
        angle = angle - 180
    return angle


def flip_image(img, metadata, file_prefix, flip_param):
    img_flip = img.copy()
    if (-2 == flip_param):
        return img_flip, list(metadata), file_prefix[:]
    height, width, channels = img.shape
    flip_metadata = list(metadata)  # copy
    img_flip = cv2.flip( img_flip, flip_param )
    caption = ""                        # caption is just the string form of the new metadata
    new_metadata = []
    for md in flip_metadata:
        # parse_txt_file gives us md =  [cx, cy, a, b, cos(2*angle), sin(2*angle), 0 (noobj=0, i.e. existence), num_rings]
        [cx, cy, a, b, angle, rings] =  md
        if (flip_param in [0,-1]):
            cy = height - cy
            angle = -angle #  sin2t *= -1       # flip the sin
        angle = cleanup_angle(angle)
        if (flip_param in [1,-1]):
            cx = width - cx
            angle = 180 - angle #cos2t *= -1       # flip the cos
        angle = cleanup_angle(angle)

        # output metadata format is md = [cx, cy,  a, b, angle, num_rings]
        new_metadata.append( [cx, cy,  a, b, angle, rings] )

    if (0==flip_param):
        new_prefix = file_prefix + "_v"
    elif (1==flip_param):
        new_prefix = file_prefix + "_h"
    else:
        new_prefix = file_prefix + "_vh"
    return img_flip, new_metadata, new_prefix


def blur_image(img, metadata, file_prefix, kernel=(3,3)):
    new_img = img.copy()
    new_img = cv2.GaussianBlur(img,kernel,0)
    new_prefix = file_prefix + "_b"
    return new_img, metadata, new_prefix


def rotate_image(img, metadata, file_prefix, rot_angle, rot_origin=None):
    # note that cv2 sometimes reverses what normal humans consider x and y coordinates
    new_img = img.copy()
    if (0 == rot_angle):
        return new_img, list(metadata), file_prefix
    height, width, channels = img.shape
    if (rot_origin is None):                            # if not specified, rotate about image center
        rot_origin = (width/2, height/2)
    rot_matrix = cv2.getRotationMatrix2D(rot_origin, rot_angle, 1.0)
    new_img = cv2.warpAffine(new_img, rot_matrix, (width,height))

    new_metadata = []
    for md in metadata:
        [cx, cy, a, b, angle, rings] =  md
        angle -= rot_angle
        angle = cleanup_angle( angle )
        myPoint = np.transpose( np.array( [ cx, cy, 1] ) )
        newPoint = np.matmul ( rot_matrix, myPoint)
        cx, cy = int(round(newPoint[0])), int(round(newPoint[1]))
        new_metadata.append( [cx, cy,  a, b, angle, rings] )

    new_prefix = file_prefix[:] + "_r{:>.2f}".format(rot_angle)
    return new_img, new_metadata, new_prefix

def invert_image(img, metadata, file_prefix):
    prefix = file_prefix +"_i"
    return cv2.bitwise_not(img.copy()), list(metadata), prefix


def augment_data(path='Train/'):
    print("augment_data: Augmenting data in Train/")

    img_file_list = sorted(glob.glob(path+'*.png'))
    txt_file_list = sorted(glob.glob(path+'*.txt'))
    assert( len(img_file_list) == len(txt_file_list))
    numfiles = len(img_file_list)
    for i in range(numfiles):
        if (0 == i % 50):
            print("     Progress:  i = ",i," of ",numfiles)
        img_filename = img_file_list[i]
        txt_filename = txt_file_list[i]

        prefix = os.path.splitext(img_filename)[0]
        #print(" img_filename = ",img_filename,", prefix = ",prefix)
        img =  cv2.imread(img_filename)
        metadata = read_metadata(txt_filename)
        '''
        # flip  [ vertical, horizontal, both v & h ]
        for flip_param in [0,1,-1]:
            img_flip, new_metadata, new_prefix = flip_image(img, metadata, prefix, flip_param)
        '''
        img_count = 0
        names = []

        for flip_param in [-2,0,1,-1]:  # flip  [ not at all, vertical, horizontal, both v & h ]
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
                    for do_inv in [False]:#  , True]:    # actually @achmorrison specifies: count rings using white/grey not black
                        if (do_inv):
                            inv_img, inv_metadata, inv_prefix =invert_image( rot_img, rot_metadata, rot_prefix)

                        names.append(inv_prefix)
                        img_count += 1
                        # Actually output the img file and the metadata text file
                        caption = caption_from_metadata( inv_metadata )
                        if (True):     # TODO: quick flag to turn off file writing
                            with open(inv_prefix+".txt", "w") as text_file:
                                text_file.write(caption)
                            cv2.imwrite(inv_prefix+'.png',inv_img)
                        #print(img_count, inv_prefix)
        unique = len(set(names))
        #print("   Generated ",img_count," images and ",unique," unique names")


if __name__ == "__main__":
    augment_data()
