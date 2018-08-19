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
from utils import *
import random
from multiprocessing import Pool
from functools import partial


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


def cleanup_angle(angle):    # not really needed, given that we use sin & cos of (2*angle) later
    while (angle < 0):
        angle += 180
    while (angle >= 180):
        angle = angle - 180
    return angle


def flip_image(img, metadata, file_prefix, flip_param):
    img_flip = img.copy()
    if (-2 == flip_param):   # do nothing
        return img_flip, list(metadata), file_prefix[:]
    height, width, channels = img.shape
    flip_metadata = list(metadata)  # copy
    img_flip = cv2.flip( img_flip, flip_param )
    caption = ""                        # caption is just the string form of the new metadata
    new_metadata = []
    for md in flip_metadata:
        # parse_meta_file gives us md =  [cx, cy, a, b, cos(2*angle), sin(2*angle), 0 (noobj=0, i.e. existence), num_rings]
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


def blur_image(img, metadata, file_prefix, kernel=(3,3)):  # unused
    new_img = img.copy()
    new_img = cv2.GaussianBlur(new_img,kernel,0)
    new_prefix = file_prefix + "_b"
    return new_img, list(metadata), new_prefix


def cutout_image(img, metadata, file_prefix, num_regions):
    """
     "Improved Regularization of Convolutional Neural Networks with Cutout", https://arxiv.org/abs/1708.04552
      All we do is chop out rectangular regions from the image, i.e. "masking out contiguous sections of the input"
       Unlike the original cutout paper, we don't cut out anything too huge, and we use random (greyscale) colors
    Note: leaves metadata unchanged
    """
    new_img = img.copy()
    if (0 == num_regions):   # do nothing
        return new_img, list(metadata), file_prefix
    height, width, channels = img.shape
    minsize, maxsize = 20, int(height/3)
    for region in range(num_regions):
        pt1 = ( np.random.randint(0,width-minsize), np.random.randint(0,height-minsize))  # upper left corner of cutout rectangle
        rwidth, rheight = np.random.randint(minsize, maxsize), np.random.randint(minsize, maxsize)  # width & height of cutout rectangle
        pt2 = ( min(pt1[0] + rwidth, width-1) , min(pt1[1] + rheight, height-1)  )   # keep rectangle bounded in image
        cval = np.random.randint(0,256)   # color value (0-255) to fill with; original cutout paper uses black
        color = (cval,cval,cval)
        cv2.rectangle(new_img, pt1, pt2, color, -1)   # -1 means filled
    new_prefix = file_prefix +"_c" + str(num_regions)  # TODO: note running twice with same num_regions will/may overwrite a file
    return new_img, list(metadata), new_prefix


def rotate_image(img, metadata, file_prefix, rot_angle, rot_origin=None):
    # note that cv2 sometimes reverses what normal humans consider x and y coordinates
    new_img = img.copy()
    if (0 == rot_angle):     # do nothing
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


def invert_image(img, metadata, file_prefix):  # unused
    prefix = file_prefix +"_i"
    return cv2.bitwise_not(img.copy()), list(metadata), prefix


def translate_image(img, metadata, file_prefix, trans_index):
    """
    translate an entire image and its metadata by a certain amount
    Note: for a 'vanilla' CNN classifer, translation should have no effect, however
          for a YOLO-style (or any?) object detector, it will/can make a difference.
    see https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    """
    new_img = img.copy()
    if (0 == trans_index):  # do nothing
        return new_img, list(metadata), file_prefix

    trans_max = 40    # max number of pixels, in any direction
    xt = int(round(trans_max * (2*np.random.random()-1) ))
    yt = int(round(trans_max * (2*np.random.random()-1) ))
    rows, cols, _ = img.shape
    M = np.float32([[1,0,xt],[0,1,yt]])
    new_img = cv2.warpAffine(new_img, M, (cols,rows))
    new_metadata = []
    for md in metadata:
        [cx, cy, a, b, angle, rings] =  md
        cx, cy = cx + xt, cy + yt
        new_metadata.append( [cx, cy,  a, b, angle, rings] )
    new_prefix = file_prefix[:] + "_t"+str(xt)+','+str(yt)
    return new_img, new_metadata, new_prefix


def augment_one_file(img_file_list, meta_file_list, n_augs, file_index):
    """
    Here is where successive augmentations of a single file (in a list) takes place
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
        # flip image
        flip_param = np.random.choice([-2,0])   # leave unchanged, or flip vertically; no other flips are relevant to this dataset
        img, metadata, prefix = flip_image(orig_img, orig_metadata, orig_prefix, flip_param)

        # rotate image
        rot_angle = np.sign(random.random()-0.5)*(1 + 5*np.random.random())
        img, metadata, prefix = rotate_image( img, metadata, prefix, rot_angle)

        # translate image
        img, metadata, prefix = translate_image( img, metadata, prefix, np.random.randint(10))

        #After all the above changes: Actually output the img file and the metadata file
        caption = caption_from_metadata( metadata )
        if (True):     # TODO: quick flag to turn off file writing (if set to False)
            with open(prefix+meta_extension, "w") as meta_file:
                meta_file.write(caption)
            cv2.imwrite(prefix+'.png', img)
    return

    '''
    # --- Old way of doing augmentation: nested for loops

    # flip image?  note for the Zooniverse steelpan dataset, horizontal flipping is irrelevant; arguably so is vertical but we'll do it
    for flip_param in [-2,0]: # ,1,-1]:  # flip  [ not at all, vertical, horizontal, both v & h ]
        flip_img, flip_metadata, flip_prefix = flip_image(img, metadata, prefix, flip_param)

        blur_img = flip_img.copy()
        blur_prefix = flip_prefix
        blur_metadata = list(flip_metadata)
        for do_blur in [False]:#  , True]:  False = Nah. Real steelpan images are already blurry enough
            if do_blur:
                blur_img, blur_metadata, blur_prefix = blur_image( blur_img, flip_metadata, flip_prefix)

            # rotate image (a little bit)?
            num_rot = 4             # number of rotated image variations to generate
            for irot in range(num_rot):
                if (0 == irot):
                    rot_angle = 0           # for the first instance, don't rotate at all
                else:
                    rot_angle = np.sign(random.random()-0.5)*(1 + 5*np.random.random())  # rotate by 1 to 6 degrees, either direction
                rot_img, rot_metadata, rot_prefix = rotate_image( blur_img, blur_metadata, blur_prefix, rot_angle)


                # cutout image?
                for cutout_param in [0]:#  Nah. Wait & do cutout on the fly [2,3,4,6]:  # various #s of cutout regions
                    co_img, co_metadata, co_prefix = cutout_image(rot_img, rot_metadata, rot_prefix, cutout_param)


                    inv_img = co_img.copy()
                    inv_metadata = list(co_metadata)
                    inv_prefix = co_prefix
                    # invert color?
                    for do_inv in [False]:#  , True]:    # actually @achmorrison specifies: count rings using white/grey not black
                        if (do_inv):
                            inv_img, inv_metadata, inv_prefix =invert_image( rot_img, rot_metadata, rot_prefix)

                        # translate image?   Note that object detection is not *just* a CNN, so translations do matter
                        for do_trans in range(3):
                            trans_img, trans_metadata, trans_prefix = translate_image( inv_img, inv_metadata, inv_prefix, do_trans)


                            #After all the above changes: Actually output the img file and the metadata file
                            caption = caption_from_metadata( trans_metadata )
                            if (True):     # TODO: quick flag to turn off file writing (if set to False)
                                with open(trans_prefix+meta_extension, "w") as text_file:
                                    text_file.write(caption)
                                cv2.imwrite(trans_prefix+'.png',trans_img)
    return
    '''

def augment_data(path='Train/', n_augs=49):
    print("augment_data: Augmenting data in",path,'by a factor of',n_augs+1)

    img_file_list = sorted(glob.glob(path+'*.png'))
    meta_file_list = sorted(glob.glob(path+'*'+meta_extension))
    assert( len(img_file_list) == len(meta_file_list))

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
    augment_data()
