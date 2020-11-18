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
from augment_preproc import *
#from gen_fake_espi import *

meta_extension = '.csv'

def copy_or_link(src, dst, link=False):
    if link:
        os.symlink(src, dst+os.path.basename(src))
    else:
        copy(src, dst)

def distribute_dataset(real_data_dir, new_dir, k=1):
    print(f"distribute_dataset: Copying data files (images & meta) from {real_data_dir} to {new_dir}: Train/, Val/...")

    # Get list of input files (images and text annotations)
    img_file_list = sorted(glob.glob(real_data_dir+'/*.png'))
    base = os.path.basename(img_file_list[0])
    meta_file_list = sorted(glob.glob(real_data_dir+'/*'+meta_extension))
    assert len(img_file_list) == len(meta_file_list),"Error, mismatch of img and meta files"

    # randomize the order of the list
    numfiles = len(img_file_list)
    print("Found",numfiles,"original data files")
    indices = list(range(numfiles))
    shuffle(indices)     # randomize order;  note that shuffle operates in place

    # make copies of things in Train & Val (& maybe Test) directories
    for dir in [new_dir, new_dir+'Train', new_dir+'Val']:  # leaving out Test b/c not enough real data
        print(f"Creating directory {dir}")
        make_sure_path_exists(dir)

    for i in range(len(indices)):
        frac = i * 1.0 / numfiles             # how far into the dataset are we
        dest = new_dir+'Train/' if frac < 0.80 else new_dir+'Val/'  # Train/Val split at 80%
        copy_or_link(img_file_list[indices[i]],  dest, link=(k>0))
        copy_or_link(meta_file_list[indices[i]], dest, link=(k>0))
    return numfiles


# --- Main code
if __name__ == "__main__":
    random.seed(1)  # for determinism. note that k-fold cross vals will be random relative to each other

    import argparse
    parser = argparse.ArgumentParser(description="Sets up real data, augments in Train/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--original', help='directory containing original data',
            default='/home/shawley/datasets/cleaned_zooniverse_steelpan/')
    parser.add_argument('--name', help='Nnme of directory for new dataset',default='.')
    parser.add_argument('-a', '--augs', type=int,
        help='number of augmentations per image to generate', default=42)
    parser.add_argument('-k', '--kfold', type=int,
        help='number of different Test/Val crossvalidation shufflings to generate', default=1)
    args = parser.parse_args()


    for k in range(args.kfold):  # generate multiple cross-validation versions of dataset (default= only one)
        if args.kfold > 1:
            print(f"\n**********************  Cross-val: k = {k+1}/{args.kfold}  *************************\n")
        # generate the data
        new_dir = args.name+f'_k{k+1}/' if (k > 0) else args.name+'/'   # don't add k-suffix for first run-through
        print(f"Clearing directories in {new_dir} (if they exist): Train/ Test/ Val/")
        os.system(f"cd {new_dir}; rm -rf Test Train Val")
        numfiles = distribute_dataset(args.original, new_dir, k=k)
        augment_data(path=new_dir+'Train/', n_augs=args.augs)        # augment in Train only
        #gen_fake_espi(numframes=numfiles*8, train_only=True)  # add synthetic data to Training set
