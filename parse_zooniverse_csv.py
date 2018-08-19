#! /usr/bin/env python3

# Reads @achmorrison et al's 'cleaned' (rather than raw) aggregated ellipse info
# Currently expects file to be called 'zooniverse_labeled_dataset.csv'.
# TODO: Should make filename a command line options

import pandas as pd
import json
import os
from shutil import copy2
import errno
import sys

meta_extension = '.csv'   # extension for output metadata files

debug = 0

def make_sure_path_exists(path):
    try:                # go ahead and try to make the the directory
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:  # ignore error if dir already exists
            raise


def create_meta_file(filename, datalist):
    if (debug > 2):
        print("   Creating metadata file ",filename,":")
    datastring = "cx,cy,a,b,angle,rings\n"
    num_an = len(datalist)           # number of antinodes
    for i in range(num_an):
        if (debug > 3):
            print(" i = ",i,", an = ",an," datalist[i] = ",datalist[i])
        datastring += "{0},{1},{2},{3},{4},{5}".format(datalist[i][0], datalist[i][1], datalist[i][2], datalist[i][3], datalist[i][4], datalist[i][5])
        if (i < num_an-1):
            datastring += "\n"

    with open(filename, "w") as meta_file:
        meta_file.write(datastring)
    return


outpath = 'parsed_zooniverze_steelpan'
inpath = 'zooniverse_steelpan'

print("Making sure directory",outpath,"exists")
make_sure_path_exists(outpath)

# Now we just append each line of data to a new metadata file for each filename
# first, as a safeguard delete any existing metadata files in outpath
print("Removing any previous files in",outpath)
import glob, os
for f in glob.glob(outpath+"/*"+meta_extension):
    os.remove(f)


in_filename = 'zooniverse_labeled_dataset.csv'
# @achmorrison:  "The order of each row is: x, y, filename, fringe_count, rx, ry, angle"
col_names = ['cx', 'cy', 'filename', 'rings', 'a', 'b', 'angle']


print("Reading metadata file",in_filename)
df = pd.read_csv(in_filename, names=col_names) # no header
df.drop_duplicates(inplace=True)  # sometimes the data from Zooniverse has duplicate rows

# df = df.sort_values(by=['filename'])   # don't need to sort
print("Writing individual metadata files")
for index, row in df.iterrows():
    #print("index, row = ",index,row)
    cx, cy = row['cx'], row['cy']
    ref_filename = row['filename']
    ref_filename = ref_filename.replace('bmp.png','png')   # "bmp.png" is confusing
    rings = row['rings']
    a, b = row['a'], row['b']
    angle = row['angle']
    # we want the semimajor axis a to be greater than the semiminor axis b
    # Enforcing this will introduce a +/- 90 degree change in the angle.
    if (b > a):
        a, b = b, a            # swap
        angle = angle + 90     # could subtract 90 instead; we're going to just take sin & cos of 2*angle later anyway

    meta_filename = os.path.splitext(ref_filename)[0]+meta_extension

    # if this is a new one then, create file   # --- and give it a header string
    meta_file_path = outpath + '/' + meta_filename
    if not os.path.exists(meta_file_path):
        print("writing to",meta_file_path)
        #with open(meta_file_path, "w") as meta_file:   NO header
        #    meta_file.write(headerstring)
        # also, copy the image itself over
        copy2(inpath+'/'+ref_filename, outpath+'/'+ref_filename)

    # add the data
    # datastring = "[{0}, {1}, {2}, {3}, {4}, {5}]".format(cx,cy,a,b,angle,rings) # Old hawley format
    datastring = "{0}, {1}, {2}, {3}, {4}, {5}".format(cx,cy,a,b,angle,rings) # CSV-appropriate format
    with open(meta_file_path, "a") as meta_file:
        meta_file.write(datastring+"\n")
