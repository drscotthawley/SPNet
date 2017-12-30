#!/usr/bin/env python3

# This automates the preparation of data prior to training.
# Here's what it does:
# 1. Move: Shuffles the real data, putting some of it in Test/, Train/ and Val/
# 2. Augments the real data in Train/
# 3. Generates synthetic data which also goes in Train/

# Assumes you have ALREADY run parse_zooniverse_csv.py and that all real data is now
# in <real_data_dir>
real_data_dir = "../datasets/parsed_zooniverze_steelpan/"

import glob
import os
from shutil import copy
from random import shuffle
from utils import *
from augment_data import *
from gen_fake_espi import *

def distribute_dataset():
    print("distribute_dataset: Copying data files (images & txt) from",real_data_dir,"to Train/, Val/ and Test/...")
    # Get list of input files (images and text annotations)
    img_file_list = sorted(glob.glob(real_data_dir+'*.png'))
    #print("img_file_list[0] = ",img_file_list[0])
    base = os.path.basename(img_file_list[0])
    #print("base = ",base)
    txt_file_list = sorted(glob.glob(real_data_dir+'*.txt'))

    # randomize the order of the list
    numfiles = len(img_file_list)
    indices = list(range(numfiles))
    shuffle(indices)     # note that shuffle works in place

    # copy things into Test, Train and Vals directories
    make_sure_path_exists('Train')
    make_sure_path_exists('Test')
    make_sure_path_exists('Val')
    for i in indices:
        frac = i * 1.0 / numfiles
        if frac < 0.6:
            dest = 'Train/'
        elif (frac > 0.8):
            dest = 'Test/'
        else:
            dest = 'Val/'
        copy(img_file_list[i], dest)
        copy(txt_file_list[i], dest)

os.system("rm -rf Test Train Val")
distribute_dataset()
#os.system("python augment_data.py")
augment_data()
#os.system("python gen_fake_data.py")
gen_fake_espi()
