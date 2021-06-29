#! /usr/bin/env python3
"""
This produces bounding boxes and "existence" objects (i.e. all of the same class = "object")
from the rotated ellipses.  For use in testing with other object detector codes.
"""
import pandas as pd
import numpy as np
import json
import os
from shutil import copy2
import errno
import sys


def get_ellipse_bb(x, y, major, minor, angle_deg):  # cx, cy, a, b, angle
    '''Get bounding box of ellipse,
    cf. https://gist.github.com/smidm/b398312a13f60c24449a2c7533877dc0
    '''
    t = np.arctan(-minor / 2 * np.tan(np.radians(angle_deg)) / (major / 2))
    [max_x, min_x] = [x + major / 2 * np.cos(t) * np.cos(np.radians(angle_deg)) -
                      minor / 2 * np.sin(t) * np.sin(np.radians(angle_deg)) for t in (t, t + np.pi)]
    t = np.arctan(minor / 2 * 1. / np.tan(np.radians(angle_deg)) / (major / 2))
    [max_y, min_y] = [y + minor / 2 * np.sin(t) * np.cos(np.radians(angle_deg)) +
                      major / 2 * np.cos(t) * np.sin(np.radians(angle_deg)) for t in (t, t + np.pi)]
    return int(min_x), int(min_y), int(max_x), int(max_y)


# Inputs:
datapath = '/home/shawley/datasets/'
#in_filename = 'zooniverse_labeled_dataset.csv'   # input CSV filename
#in_filename = datapath+'average_all-good-ellipses-five_or_more-072420.csv' # input CSV filename
in_filename = datapath+'average_good_ellipses_bad_values_removed.csv' # input CSV filename
imgpath = datapath+'zooniverse_steelpan/'                    # directory where ALL images are stored (e.g. in lecun:datasets/+this)

# Outputs:
out_filename = datapath+'zooniverse_bounding_boxes.csv'

# @achmorrison:  "The order of each row is: x, y, filename, fringe_count, rx, ry, angle"
col_names = ['cx', 'cy', 'filename', 'rings', 'a', 'b', 'angle']

print("Reading metadata file",in_filename)
df = pd.read_csv(in_filename, names=col_names) # no header
df.drop_duplicates(inplace=True)  # sometimes the data from Zooniverse has duplicate rows
df.dropna(inplace=True)           # drop rows containing NaNs
df = df[(df[['rings']] != 0).all(axis=1)]  # drop rows where ring count is zero
n = df.shape[0] # len(df.index) # number rows

# Convert to bboxes
boxinfo = [get_ellipse_bb(row[0],row[1],row[2],row[3],row[4]) for row in zip(df['cx'],df['cy'],df['a'],df['b'],df['angle'])]
#print("boxinfo = ",boxinfo)
box_df = pd.DataFrame(boxinfo, columns=['xmin', 'ymin', 'xmax', 'ymax'])

# new dataframe, following format of https://airctic.com/0.7.0/custom_parser/
new_df = pd.concat([df, box_df], axis=1)
new_df['width'] = [512]*n
new_df['height'] = [384]*n
new_df['label'] = ['object']*n

new_col_names = ['filename','width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
new_df = new_df[new_col_names]
print("new_df = ",new_df)

print("Writing to new file",out_filename)
new_df.to_csv(out_filename, index=False)
