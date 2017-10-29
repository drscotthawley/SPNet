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
import errno
import time
import multiprocessing
from utils import *

winName = 'ImgWindowName'
#imWidth = 224                  # needed for MobileNet
#imHeight = imWidth
imWidth = 512
imHeight = 384


# Define some colors: openCV uses BGR instead of RGB
blue = (255,0,0)
red = (0,0,255)
green = (0,255,0)
white = (255)
black = (0)
grey = (128)


def draw_waves(img):
    #pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    xs = np.arange(0, imWidth)
    ys = np.arange(0, imHeight)
    #xcoords = np.tile(xs, (img.shape[1],1))
    #ycoords = np.tile(ys, (img.shape[0],1))

    amp = random.randint(10,200)
    x_wavelength = random.randint(100,imWidth/2)
    y_wavelength = random.randint(20,int(imHeight/4))
    thickness = random.randint(3,10)
    slope = 3*(np.random.rand()-.5)
    numlines = 40+int(imHeight/y_wavelength)
    #print(amp,x_wavelength,y_wavelength,thickness,slope,numlines)
    #img = img+np.array( amp* np.cos( xcoords/x_wavelength)*np.sin(ycoords/y_wavelength), dtype=np.uint8)
    #return

    for j in range(numlines):
        y_start = j*y_wavelength - img.shape[1]*abs(slope)
        #print("y_start = ",y_start)
        pts = []
        for i in range(len(xs)):
            pt = [ int(xs[i]), int(y_start + slope*xs[i]+ amp * np.cos(xs[i]/x_wavelength))]
            pts.append(pt)
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], False, grey, thickness=thickness)
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
    thickness = round(min(axes)/(num_wbrings))
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
    #print("    does_overlap_previous: box = ",box)
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
        caption = "[{0}, {1}, {2}, {3}, {4}, {5}]".format( imWidth/2.0,  imHeight/2.0,    0,  imWidth/4.0,    imHeight/4.0,    90.0)
    for an in range(num_antinodes): # draw a bunch of antinodes
        num_rings = random.randint(1,11)            # well say that an antinode has at least 1 ring

        axes = (random.randint(15,int(imWidth/3)), random.randint(15,int(imHeight/3)))   # semimajor and semiminor axes of ellipse
        axes = sorted(axes, reverse=True)   # do descending order, for definiteness

        center = (random.randint(axes[0],imWidth-axes[0]),
            random.randint(axes[1],imHeight-axes[1]))
        angle = random.randint(1,179)       # ellipses are symmetric after 180 degree rotation
        box = get_ellipse_box(center,axes, angle)
        # make sure they don't overlap, and are in bounds of image
        while(True == does_overlap_previous(box, boxes_arr)
            or (box[0]<0) or (box[2] > imWidth)
            or (box[1]<0) or (box[3] > imHeight)  ):
            # generate new values
            #print("    Re-doing it")
            axes = (random.randint(20,int(imWidth/3)), random.randint(20,int(imHeight/3)))
            axes = sorted(axes, reverse=True)   # do descending order
            center = (random.randint(axes[0],imWidth-axes[0]),
                random.randint(axes[1],imHeight-axes[1]))
            angle = random.randint(1,180)
            box = get_ellipse_box(center,axes, angle)

        draw_rings(img,center,axes,angle=angle,num_rings=num_rings)

# [x coordinate of antinode center, y coordinate, subject_id/frame_num,
# number of fringes for that region, radius in the x direction, radius in y (if there were no rotation)
# and the angle the ellipse is at]
        this_caption = "[{0}, {1}, {2}, {3}, {4}, {5}]".format(center[0], center[1],num_rings,axes[0],axes[1],angle)
        #print(this_caption)
        if (an > 0):
            caption+="\n"
        caption += this_caption
        boxes_arr.append(box)
    return img, caption


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# TODO: haven't figured out how to pass args when multiprocessing; the following globals should be replaced w/ args at some point
frame_start = 0
num_frames = 240000
num_tasks = 12    # we've got 12 processors. choose a number with a large (num_tasks % 12) that's still a factor of num_frames
frames_per_task= int(round(num_frames / num_tasks))

def gen_images(task):
    # have different tasks generate different parts of the dataset
    val = task*1.0/num_tasks
    if (val < 0.6):     # 3 * 20% = 60% Train
        dirname = 'Train'
    elif (val >= 0.8):    # 20%  Test
        dirname = 'Test'
    else:
        dirname = 'Val'    # 20% Val

    for iframe in range(frames_per_task):
        framenum = frame_start + task * frames_per_task + iframe
        if (0 == framenum % 50):
            pad = ' '.rjust(4*task)
            print(pad,"task",task,": framenum = ",framenum,", ending at ",(task+1)*frames_per_task-1)
        np_dims = (imHeight, imWidth, 1)                       # for numpy, dimensions are reversed

        img = np.zeros(np_dims, np.uint8)

        draw_waves(img)

        max_antinodes = 6
        num_antinodes= 6# random.randint(0,max_antinodes)   # should we allow zero antinodes?

        img, caption = draw_antinodes(img, num_antinodes=num_antinodes)
        #img = cv2.GaussianBlur(img,(7,7),0)
        noise = cv2.randn(np.zeros(np_dims, np.uint8),50,50);
        img = cv2.add(img, noise)

        prefix = dirname+'/steelpan_'+str(framenum).zfill(7)
        #print("Task ", task,": writing with prefix",prefix)
        cv2.imwrite(prefix+'.bmp',img)
        with open(prefix+".txt", "w") as text_file:
            text_file.write(caption)



# --- Main code
if __name__ == "__main__":
    start_time = time.clock()

    make_sure_path_exists('Train')
    make_sure_path_exists('Val')
    make_sure_path_exists('Test')

    random.seed(1)



    num_procs = multiprocessing.cpu_count()
    print(num_procs," processors available, assigning ",num_tasks,"tasks..")
    tasks = range(num_tasks)
    pool = multiprocessing.Pool()
    results = pool.map(gen_images, tasks)
    pool.close()
    pool.join()


    print(time.clock() - start_time, "seconds")
#cv2.destroyAllWindows()
