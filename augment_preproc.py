#! /usr/bin/env python3

# Implements data-augmentation in the form of flipping images vertically, horizontally, and "both"
# when images are altered, metadata (in accompanying text files) also needs to be altered
# By default, only affects *Training* data.  Leaves Val & Test alone

# TODO: Create a generator to be used within Keras fit routine, move cutout &
#       other routines that don't affect metadata there

import numpy as np
import cv2
import glob
import os
import sys
from spnet.utils import *
from spnet.augmentation import *
import random
from multiprocessing import Pool
from functools import partial
from keras.callbacks import Callback

meta_extension = ".csv"


def read_metadata(meta_filename):
    col_names = ['cx', 'cy', 'a', 'b', 'angle', 'rings']
    df = pd.read_csv(meta_filename,header=None,names=col_names)  # read metadata file
    df.drop_duplicates(inplace=True)  # sometimes the data from Zooniverse has duplicate rows

    arrs = []    # this is a list of lists, containing all the ellipse info & ring counts for an image
    for index, row in df.iterrows() :
        cx, cy = row['cx'], row['cy']
        a, b = row['a'], row['b']
        angle, num_rings = float(row['angle']), row['rings']
        tmp_arr = [cx, cy, a, b, angle, num_rings]
        arrs.append(tmp_arr)
    arrs = sorted(arrs,key=itemgetter(0,1))     # sort by y first, then by x
    return arrs


def caption_from_metadata(metadata):
    """
    converts metadata list-of-lists to caption string which is one antinode per line
    """
    caption = ""
    for an in range(len(metadata)):
        [cx, cy, a, b, angle, rings] = metadata[an]
        #this_caption = "[{0}, {1}, {2}, {3}, {4}, {5}]".format(cx, cy, a, b, angle, rings)
        this_caption = "{0},{1},{2},{3},{4},{5}".format(cx, cy, a, b, angle, rings)
        if (an > 0):
            caption +="\n"
        caption += this_caption
    return caption


def augment_one_file(img_file_list, meta_file_list, n_augs, file_index, bp_too=False):
    """
    Here is where successive augmentations of a single file (in a list) takes place

    Currently, we only rotate and/or translate.  Further augmentations adding cutout,
    noise and/or flipping the image now happens 'on the fly' via the
    AugmentOnTheFly() callback in train_spnet.py
    """
    i = file_index
    if (0 == i % 10):
        print("     Progress: i =",i,"/",len(img_file_list))
    img_filename = img_file_list[i]
    meta_filename = meta_file_list[i]

    orig_prefix = os.path.splitext(img_filename)[0]
    orig_img =  cv2.imread(img_filename)
    orig_metadata = read_metadata(meta_filename)

    for aug in range(n_augs):
        #print("aug, orig_prefix =",aug, orig_prefix)
        img, metadata, prefix = orig_img, orig_metadata, orig_prefix

        #flip image
        flip_param = np.random.choice([-2,-1,0,1])
        img, metadata, prefix = flip_image(orig_img, orig_metadata, orig_prefix, flip_param)

        # rotate image +/- by some small random angle
        rot_max = 20   # degrees
        rot_angle = np.random.uniform(-rot_max, high=rot_max)
        img, metadata, prefix = rotate_image( img, metadata, prefix, rot_angle)

        # translate image
        img, metadata, prefix = translate_image( img, metadata, prefix, np.random.randint(10))

        # bandpass mixup?
        if bp_too:
            img = bandpass_mixup(img)

        #After all the above changes: Actually output the img file and the metadata file
        caption = caption_from_metadata( metadata )
        if (True):     # TODO: quick flag to turn off file writing (if set to False)
            with open(prefix+meta_extension, "w") as meta_file:
                meta_file.write(caption)
            cv2.imwrite(prefix+'.png', img)
    return


def augment_data(path='Train', n_augs=39):
    print("augment_data: Augmenting data in",path,'by a factor of',n_augs+1)
    path += '/'
    img_file_list = sorted(glob.glob(path+'*.png'))
    meta_file_list = sorted(glob.glob(path+'*'+meta_extension))
    assert len(img_file_list) == len(meta_file_list), f"{len(img_file_list)} images, {len(meta_file_list)} CSV files. Should be the same number"

    # here's where you parallelize
    file_indices = tuple( range(len(img_file_list)) )
    numfiles = len(img_file_list)
    print("Found",numfiles,"files in",path)

    cpu_count = os.cpu_count()
    print("Mapping augmentation operation across",cpu_count,"processes")
    pool = Pool(cpu_count)
    pool.map(partial(augment_one_file, img_file_list, meta_file_list, n_augs), file_indices)
    new_numfiles = len(sorted(glob.glob(path+'*.png')))
    print("Augmented from",numfiles,"files up to",new_numfiles)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="augments data in path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--datapath', #type=argparse.string,
        help='dataset directory in which to augment', default="Train/")
    parser.add_argument('-n', '--naugs', type=int, help='number of augmentations per image to generate', default=42)
    args = parser.parse_args()

    augment_data(path=args.datapath, n_augs=args.naugs)
