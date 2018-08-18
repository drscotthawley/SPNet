#! /usr/bin/env python3

# Generates 'fake' images akin to the ESPI images of steelpan drums, from
# https://www.zooniverse.org/projects/achmorrison/steelpan-vibrations
#

# num_rings can be between 0 and 11.  0 means there is no antinode there
# You can also have num_antinodes=0, but this produces more of a 'skewed' dataset
#   than if you use num_rings=0 instead and let num_antinodes=1

# note that this doesn't require any GPU usage, just a lot of disk usage

# Added multiprocesing: runs lots of processes to cut execution time down

import numpy as np
import cv2
import random
import os
import time
import multiprocessing as mp
from utils import *
import sys, traceback
from shutil import get_terminal_size

winName = 'ImgWindowName'
#imWidth = 224                  # needed for MobileNet
#imHeight = imWidth
imWidth = 512
imHeight = 384

meta_extension = ".csv"     # file extension for metadata files

# Define some colors: openCV uses BGR instead of RGB
blue = (255,0,0)
red = (0,0,255)
green = (0,255,0)
white = (255)
black = (0)
grey = (128)

blur_prob = 0.9    # probability that an image gets blurred

# TODO: haven't figured out how to pass args when multiprocessing; the following globals should be replaced w/ args at some point
# for now, we define them globally but set them in __main__
# TODO: actually, can use partial() to pass args.  Need to update this.
frame_start = 0
num_frames = 0
num_tasks = 0
frames_per_task= 0

train_only_global = True           # Only gen fake images for the training set. False= make Test & Val images too
pad = ''


def blur_image(img, kernel_size=7):
    if (0==kernel_size):
        return img
    new_img = img#.copy()
    new_img = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    return new_img


def draw_waves(img):
    #pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    xs = np.arange(0, imWidth)
    ys = np.arange(0, imHeight)

    amp = random.randint(10,200)
    x_wavelength = random.randint(100,int(imWidth/2))
    thickness = random.randint(15,40)
    slope = 3*(np.random.rand()-.5)
    y_spacing = random.randint(thickness + thickness*int(np.abs(1.5*slope)), int(imHeight/3))
    numlines = 60+int(imHeight/y_spacing)

    for j in range(numlines):   # skips y_spacing in between drawings
        y_start = j*y_spacing - img.shape[1]*abs(slope)
        pts = []
        for i in range(len(xs)):
            pt = [ int(xs[i]), int(y_start + slope*xs[i]+ amp * np.cos(xs[i]/x_wavelength))]
            pts.append(pt)
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], False, black, thickness=thickness)
    return



def swap(a,b):
    tmp = a
    a = b
    b = tmp
    return a,b

def get_ellipse_box(center, axes, angle):  # converts ellipse to bounding box
    rad = np.radians(angle)
    a = axes[0]
    b = axes[1]
    #print("get_ellipse_box: center, axes, angle = ",center,axes,angle)
    delta_x = np.sqrt(a**2 * np.cos(rad)**2 + b**2 * np.sin(rad)**2 )
    delta_y = np.sqrt(a**2 * np.sin(rad)**2 + b**2 * np.cos(rad)**2 )
    xmin = center[0] - delta_x
    ymin = center[1] - delta_y
    xmax = center[0] + delta_x
    ymax = center[1] + delta_y
    if (xmin > xmax):
        xmin,xmax = swap(xmin,xmax)
    if (ymin > ymax):
        ymin, ymax = swap(ymin,ymax)
    return [xmin,ymin,xmax,ymax]


def draw_rings(img,center,axes,angle=45,num_rings=5):
    num_wbrings = 2*num_rings  # draw in white & black
    if (0==num_wbrings):
        num_wbrings = 1      # sorry, gotta avoid any & all errors because MP is a pain to debug
    thickness = int(round( min(axes)/(num_wbrings) ))
    for j in range(num_wbrings):
        if (0 == j % 2):
            color = black
        else:
            color = grey
        thisring_axes = [axes[i] * (j+1)*1.0/(num_wbrings+1) for i in range(len(axes))]
        ellipse = draw_ellipse(img,center,thisring_axes,angle,color=color, thickness=thickness)
    return ellipse   # returns outermost ellipse

def does_overlap( a, b):
    if (a[2] < b[0]):
        #print("     a is left of b")
        return False # a is left of b
    if (a[0] > b[2]):
        #print("     a is right of b")
        return False # a is right of b
    if (a[3] < b[1]):
        #print("     a is above b")
        return False # a is above b
    if (a[1] > b[3]):
        #print("     a is below b")
        return False # a is below b
    return True

def does_overlap_previous(box, boxes_arr):
    # returns true if bounding box of new ellipse (ignoring angle) would
    # overlap with previous ellipses
    if ([] == boxes_arr):
        return False
    for i in range(len(boxes_arr)):
        #print("                boxes_arr[",i,"] = ",boxes_arr[i])
        if (does_overlap( box, boxes_arr[i])):
            return True
    return False




def draw_antinodes(img,num_antinodes=1):
    boxes_arr = []
    caption = ""

    if (num_antinodes==0):
        #caption = "[{0}, {1}, {2}, {3}, {4}, {5}]".format( imWidth/2.0,  imHeight/2.0,    0,  imWidth/4.0,    imHeight/4.0,    90.0)
        caption = "{0},{1},{2},{3},{4},{5}".format( 0,  0,    0,  0,    0,   0.0)  # as per @achmorrison's format

    for an in range(num_antinodes): # draw a bunch of antinodes

        axes = (random.randint(15,int(imWidth/3.5)), random.randint(15,int(imHeight/3.5)))   # semimajor and semiminor axes of ellipse
        axes = sorted(axes, reverse=True)   # do descending order, for definiteness


        # TODO: based on viewing real images: for small antinodes, number of rings should also be small
        num_rings = random.randint(1, 11)            # well say that an antinode has at least 1 ring


        center = (random.randint(axes[0], imWidth-axes[0]),
            random.randint(axes[1], imHeight-axes[1]))
        angle = random.randint(1, 179)       # ellipses are symmetric after 180 degree rotation
        box = get_ellipse_box(center, axes, angle)

        # make sure they don't overlap, and are in bounds of image
        # TODO: the following random placement is painfully inefficient
        trycount, maxtries = 0, 200000
        while (   ( (True == does_overlap_previous(box, boxes_arr))
            or (box[0]<0) or (box[2] > imWidth)
            or (box[1]<0) or (box[3] > imHeight)  ) and (trycount < maxtries) ):
            trycount += 1
            # if there's a problem, then generate new values - "Re-do"
            axes = (random.randint(25, int(imWidth/3)), random.randint(25, int(imHeight/3)))
            axes = sorted(axes, reverse=True)   # do descending order
            center = (random.randint(axes[0], imWidth-axes[0]),
                random.randint(axes[1], imHeight-axes[1]))
            angle = random.randint(1, 180)
            box = get_ellipse_box(center, axes, angle)

        success = False
        if (trycount < maxtries):
            draw_rings(img, center, axes, angle=angle, num_rings=num_rings)
            #this_caption = "[{0}, {1}, {2}, {3}, {4}, {5}]".format(center[0], center[1],axes[0], axes[1], angle, num_rings)
            this_caption = "{0},{1},{2},{3},{4},{5}".format(center[0], center[1],axes[0], axes[1], angle, num_rings)
            success = True
        else:   # just skip this antinode
            print("\n\r",pad,":WARNING Can't fit an=",an,"\n",sep="",end="\r")
            this_caption = ""

        if (success):               # don't add blank lines, only add lines for success
            if (an > 0):
                caption+="\n"
            caption += this_caption
            boxes_arr.append(box)
    return img, caption



def gen_images_wrapper(task):
    try:
        gen_images(task)
    except:
        print("Error on task",task,":",traceback.format_exc())


def gen_images(task):
    global pad
    if (train_only_global):
        dirname = 'Train/'
    else:
        # have different tasks generate different parts of the dataset
        val = task*1.0/num_tasks
        if (val < 0.6):     # 3 * 20% = 60% Train
            dirname = 'Train'
        elif (val >= 0.8):    # 20%  Test
            dirname = 'Test'
        else:
            dirname = 'Val'    # 20% Val

    # used for spacing out task output
    pad_space = max(14, int(round( get_terminal_size().columns / (num_tasks+0.5))))
    pad = '\033['+str(pad_space*task)+'C'   # ANSI code to move cursor to the right
    if (0==task):
        pad = ''
    pad = pad + str(task)

    task_maxframe = (task+1)*frames_per_task-1

    for iframe in range(frames_per_task):
        framenum = frame_start + task * frames_per_task + iframe
        if (0 == framenum % 1):
            print(pad,":",framenum,"/",task_maxframe,sep="",end="\r")
        np_dims = (imHeight, imWidth, 1)                       # for numpy, dimensions are reversed

        img = 128*np.ones(np_dims, np.uint8)

        #print(pad,":",framenum," DW   ",sep="",end="\r")
        draw_waves(img)   # this is the main bottleneck, execution-time-wise
        #print(pad,":",framenum," BDW ",sep="",end="\r")

        max_antinodes = 6
        num_antinodes= random.randint(0,max_antinodes)

        img, caption = draw_antinodes(img, num_antinodes=num_antinodes)

        blur_dice_roll = np.random.random()
        if (blur_dice_roll <= blur_prob):
            blur_ksize = random.choice([3,5,7])
            img = blur_image(img, kernel_size=blur_ksize)

        # post-blur noise
        noise = cv2.randn(np.zeros(np_dims, np.uint8),50,50);
        img = cv2.add(img, noise)

        prefix = dirname+'/steelpan_'+str(framenum).zfill(7)
        cv2.imwrite(prefix+'.png',img)
        with open(prefix+meta_extension, "w") as text_file:
            text_file.write(caption)

    print("\r",pad,":Finished   ",sep="",end="\r")


def gen_fake_espi(numframes=1000, train_only=True):
    global frame_start, num_frames, num_tasks, frames_per_task, train_only_global
    print("gen_fake_data: Generating synthetic data")
    num_frames = numframes
    frame_start = 0
    num_tasks = 10    # we've got 12 processors. but 10 is 'cleaner'
    frames_per_task= int(round(num_frames / num_tasks))
    train_only_global = train_only

    start_time = time.clock()

    make_sure_path_exists('Train')
    make_sure_path_exists('Val')
    make_sure_path_exists('Test')


    num_procs = mp.cpu_count()
    print(num_procs,"processors available.")
    print("Assigning",num_tasks,"parallel tasks, processing",frames_per_task,"frames each.")
    tasks = range(num_tasks)
    pool = mp.Pool()
    mp.log_to_stderr()
    print("Task progress...  <task>:<frame>/<maxframe>")
    results = pool.map(gen_images_wrapper, tasks)
    print("")
    print("Back from pool.map, results =",results)
    pool.close()
    pool.join()
    print("Back from pool.join")

    print(time.clock() - start_time, "seconds")


# --- Main code
if __name__ == "__main__":
    #global frame_start, num_frames, num_tasks, frames_per_task
    random.seed(1)
    np.random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description="trains network on training dataset")
    parser.add_argument('-n', '--numframes', type=int, help='Number of images to generate', default=500)
    parser.add_argument('-a', '--all',
        help='general all data (Val & Test too), default is Train only', default=False, action='store_true')
    args = parser.parse_args()

    gen_fake_espi(numframes=args.numframes, train_only=(not args.all))
#cv2.destroyAllWindows()
