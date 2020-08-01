#!/usr/bin/env python3

# This automates the preparation of data prior to training.
# Here's what it does:
# 1. Move: Shuffles the real data, putting some of it in Test/, Train/ and Val/
# 2. Augments the real data in Train/
# 3. Generates synthetic data which also goes in Train/

# Assumes you have ALREADY run parse_zooniverse_csv.py and that all real data is now
# in <real_data_dir>

import glob
import os
from shutil import copy
from random import seed, shuffle
from spnet.utils import *
from augment_data import *
#from gen_fake_espi import *

meta_extension = '.csv'

def distribute_dataset(real_data_dir):
    print("distribute_dataset: Copying data files (images & meta) from",real_data_dir,"to Train/, Val/ and Test/...")
    # Get list of input files (images and text annotations)
    img_file_list = sorted(glob.glob(real_data_dir+'/*.png'))
    #print("img_file_list[0] = ",img_file_list[0])
    base = os.path.basename(img_file_list[0])
    #print("base = ",base)
    meta_file_list = sorted(glob.glob(real_data_dir+'/*'+meta_extension))

    assert len(img_file_list) == len(meta_file_list),"Error, mismatch of img and meta files"

    # randomize the order of the list
    numfiles = len(img_file_list)
    print("Found",numfiles,"original data files")
    indices = list(range(numfiles))
    shuffle(indices)     # randomize order;  note that shuffle works in place

    # copy things into Test, Train and Vals directories
    make_sure_path_exists('Train')
    #make_sure_path_exists('Test')  # skip Test
    make_sure_path_exists('Val')
    for i in range(len(indices)):
        frac = i * 1.0 / numfiles
        if frac < 0.80:
            dest = 'Train/'
        else:
            dest = 'Val/'
        #else:
        #    dest = 'Val/'
        copy(img_file_list[indices[i]], dest)
        copy(meta_file_list[indices[i]], dest)
    return numfiles


# --- Main code
if __name__ == "__main__":
    random.seed(1)  # for determinism
    import argparse
    parser = argparse.ArgumentParser(description="Sets up real data, augments in Train/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', help='directory containing labeled data',
            default='/home/shawley/datasets/cleaned_zooniverse_steelpan/')
    parser.add_argument('-n', '--naugs', type=int,
        help='number of augmentations per image to generate', default=42)
    args = parser.parse_args()

    print("Clearing directories Train/ Test/ Val/")
    os.system("rm -rf Test Train Val")
    numfiles = distribute_dataset(args.data)
    augment_data(n_augs=args.naugs)
    #gen_fake_espi(numframes=numfiles*8, train_only=True)  # add synthetic data to Training set
