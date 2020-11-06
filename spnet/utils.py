import numpy as np
import pandas as pd
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import Callback
import cv2
import PIL
from operator import itemgetter
import keras.backend as K
import tensorflow as tf
import os
import errno
import random
import time
from spnet import models, diagnostics
import spnet.config as cf

# for parallelism:
from numba import jit

from multiprocessing import Pool, sharedctypes, cpu_count, get_context
from functools import partial
import gc



def make_sure_path_exists(path):
    try:                # go ahead and try to make the the directory
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:  # ignore error if dir already exists
            raise


def draw_ellipse(
        img, center, axes, angle,
        startAngle=0, endAngle=360, color=(0) ,
        thickness=2, lineType=cv2.LINE_AA, shift=10):
    # startAngle & endAngle should are arc-angles. They should stay at 0 & 360 for full ellipses...
    # TODO: grabbed this code from...? not sure what shift does
    center = (
        int(round(center[0] * 2**shift)),
        int(round(center[1] * 2**shift))
    )
    axes = (
        int(round(axes[0] * 2**shift)),
        int(round(axes[1] * 2**shift))
    )
    ellipse = cv2.ellipse(
        img, center, axes, -angle,   # -angle because the web interface is "upside down"
        startAngle, endAngle, color,
        thickness, lineType, shift)
    return ellipse


def cleanup_antinode_vars(Y_subarr):
    '''This is just a 'clean'-up wrapper to make show_pred_ellipses (below) read easier
       Y_subarr is one set of variables, i.e. Y_subarr = Y[j,an*cf.vars_per_pred:(an+1)*cf.vars_per_pred]
    '''
    [cx, cy, a, b, cos2t, sin2t, noobj, rings] = Y_subarr
    [cx, cy, a, b, noobj] = [int(round(x)) for x in [cx, cy,  a, b, noobj]] # OpenCV wants ints or it barfs
    angle = np.rad2deg( np.arctan2(sin2t,cos2t)/2.0 )  # note that we didn't cos2t and sin2t are floats, not ints
    angle = angle if angle > 0 else angle+180
    return cx, cy, a, b, angle, noobj, rings


def show_pred_ellipses(Yt, Yp, file_list, num_draw=40, log_dir='./logs/', ind_extra=None, out_csv=None, show_true=True,verbosity=0):
    """
    draws a bunch of sample output files showing where ellipses are on images

    out_csv:  filename to make a (single) zooniverse-style csv of predicted info
    """
    # Yt & Yp are already de-normed.  Yt = true,  Yp= predict
    m = Yt.shape[0]
    num_draw = min(num_draw, m, len(file_list))

    if (out_csv is not None):               # clear the output log csv file
        with open(out_csv, "w") as file_csv_out:
            file_csv_out.write('')

    if verbosity > 0: print("  format for drawing: [cx,  cy,  a,  b, angle, noobj, rings]")
    for count in range(num_draw):     # loop over all files
        j = count
        if (count >= num_draw):
            j = ind_extra          # extra plot that we specify, e.g. the worst prediction so far
        in_filename = file_list[j]

        csv_str = ''

        out_filename = log_dir+'/steelpan_pred_'+str(j).zfill(5)+'.png'
        if verbosity > 0: print("    Drawing on image",count,"of",num_draw,":",in_filename,", writing to",out_filename)
        img = load_img(in_filename)                 # this is a PIL image
        img_dims = img_to_array(img).shape

        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)      # convert from PIL to OpenCV
        img = opencvImage

        todraw_list =  [{'name':'True','Y':Yt, 'color':cf.truecolor, 'bg':cf.black,'yo':0}] if show_true else []
        todraw_list += [{'name':'Pred','Y':Yp, 'color':cf.predcolor, 'bg':cf.lightgrey, 'yo':27}]

        max_pred_antinodes = int(Yt[0].size / cf.vars_per_pred)
        for an in range(max_pred_antinodes):    # count through all antinodes
            # why two 'todraw' loops intead of just one?  Because it "looks better" if the number-text for the True doesn't get overdrawn
            #    by ellipse of the Prediction

            # first draw the ellipses
            for todraw in todraw_list:
                cx, cy, a, b, angle, noobj, rings = cleanup_antinode_vars(todraw['Y'][j,an*cf.vars_per_pred:(an+1)*cf.vars_per_pred])
                if (noobj==0):  # noobj==0 means there is an object, note int(noobj) above means it's 1 or zero here
                    if (an < 6) and (verbosity > 0):  # little text output for logging
                        print(f"          {todraw['name']}:   {cx: >4d}, {cy: >4d},  {a: >4d}, {b: >4d},   {angle: >6.2f},  {noobj: >2d}, {rings: >4.1f}")
                    if (rings > 0) and (a>=0) and (b>=0):  # only draw if you should and if you can
                        draw_ellipse(img, [cx, cy], [a,b], angle, color=todraw['color'], thickness=3)

            # now draw the text
            for todraw in todraw_list:
                cx, cy, a, b, angle, noobj, rings = cleanup_antinode_vars(todraw['Y'][j,an*cf.vars_per_pred:(an+1)*cf.vars_per_pred])
                if (noobj==0) and (rings > 0) and (a>=0) and (b>=0):
                    cv2.putText(img, "{: >3.1f}".format(rings), (cx-12,cy+todraw['yo']), cv2.FONT_HERSHEY_TRIPLEX, 0.95, color=todraw['bg'], thickness=2,  lineType=cv2.LINE_AA);  # add a little outline for readibility
                    cv2.putText(img, "{: >3.1f}".format(rings), (cx-10,cy+2+todraw['yo']), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.95, color=todraw['bg'], thickness=2, lineType=cv2.LINE_AA)  # add a little shadow/outline for readibility
                    cv2.putText(img, "{: >3.1f}".format(rings), (cx-10,cy+todraw['yo']), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=todraw['color'], thickness=1, lineType=cv2.LINE_AA)
                    if (todraw['name'] == 'Pred') and (out_csv is not None):
                        csv_str += "{},{},{},{},{},{},{}".format(cx, cy, os.path.basename(in_filename), rings, a, b, angle)+'\n'

        if (out_csv is not None) and (csv_str == ''):   # if nothing for this image, output zeros
            csv_str = '0,0,'+os.path.basename(in_filename)+',0,0,0,0\n'

        # write the original filename on the bottom left of the image, to make it easy to find again.
        cv2.putText(img, os.path.basename(in_filename), (7, orig_img_dims[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cf.black, lineType=cv2.LINE_AA); # display the filename
        cv2.putText(img, os.path.basename(in_filename), (5, orig_img_dims[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cf.white, lineType=cv2.LINE_AA); # display the filename

        cv2.imwrite(out_filename,img)           # save output image

        if (out_csv is not None):               # add to the csv log
            with open(out_csv, "a") as file_csv_out:
                file_csv_out.write(csv_str)
    return


orig_img_dims=[512,384]
means = []
ranges = []

def setup_means_and_ranges(pred_shape):
    """
    Assigns means and ranges (variances) across grid of predictors
    """
    global means, ranges

    # Assign Defaults ---------------------------------
    [cx_min, cy_min] =  [40,  40]
    [cx_max, cy_max] =[ 470, 350]
    xbinsize = int( (cx_max - cx_min) / pred_shape[0])
    ybinsize = int( (cy_max - cy_min) / pred_shape[1])

    # Also, set up the means & ranges for normalization, according to the grid of predictors
    gridmeans = np.zeros(pred_shape,dtype=cf.dtype)
    gridranges = np.zeros(pred_shape,dtype=cf.dtype)
    griddefaults = np.zeros(pred_shape,dtype=cf.dtype)
    for i in range(pred_shape[0]):
        for j in range(pred_shape[1]):
            grid_cx = i*xbinsize + cx_min + xbinsize/2
            grid_cy = j*ybinsize + cy_min + ybinsize/2
            #           format: [cx,         cy,           a,              b,     cos2theta, sin2theta,  noobj,  num_rings]   noobj = 0/1 flag for background
            griddefaults[i,j] = [grid_cx,    grid_cy,   xbinsize/2,    ybinsize/2,      -1,      0,         1,       0] # default for 'blanks'.  So noobj=1 (nothing there),  rings=0, angle is 90 degrees
            gridmeans[i,j] =    [grid_cx,    grid_cy,   xbinsize/2,    ybinsize/2,       0,      0,         0,       5] #  for noobj (=[0..1]), the means is technically 0.5, not 0.
                                                                                                                        # but we use zero because in the norm_Y routine (below), we want to keep it [0..1] and not [-0.5..0.5]
                                                                                                                        # to retain probabilistic interpretation, e.g. for sigmoid activation.   Recall these are OUTPUTS, not inputs
            gridranges[i,j] =   [xbinsize,  ybinsize,   xbinsize,      ybinsize,         2,      2,         1,      10] #+ [1]*(num_classes+1),  -1 to 1 is range of 2

    gridYi = np.copy(griddefaults)              # initialize a single grid-Y output with default values

    means = gridmeans.flatten()                 # assign global means & ranges for later
    ranges = gridranges.flatten()

    return cx_min, cy_min, cx_max, cy_max, xbinsize, ybinsize, gridYi


def norm_Y(Y, set_means_ranges=False):  # not using this yet, but might be handy
    global means, ranges
    if (False):  # this doesn't work. TODO: fix it!  For now, means & ranges are set below in true_to_pred_grid()
        means = np.mean(Y,axis=0)
        ranges = np.var(Y,axis=0)    # variance
    return (Y-means)/ranges #,  means, ranges

def denorm_Y(normY):
    global means, ranges
    return normY*ranges + means


def true_to_pred_grid(true_arr, pred_shape, num_classes=11, img_filename=None):
    """
    the essence of the YOLO-style approach
     this takes our 'true' antinode info for one image, and assigns it across the 'grid' of predictors, i.e. YOLO-style
     true_arr is a list of antinode data which has been read from a metadata file
     pred_shape has dimensions [nx, nx, preds_per_cell, vars_per_pred]
     TODO: and each value is organized (according to loss type) as
       cx, cy, a, b, angle, prob_exist, ring_count
       ... 17 variables in all
    """
    global means, ranges

    cx_min, cy_min, cx_max, cy_max, xbinsize, ybinsize, gridYi = setup_means_and_ranges(pred_shape)

    # Assign True Values -----------------------------
    assigned_counts = np.zeros(gridYi.shape[0:2],dtype=np.int)   # count up how many times a given array has been assigned
    this_shape = true_arr.shape
    for an in range(true_arr.shape[0]):
        ind_x = int((true_arr[an,0]  - cx_min) / xbinsize)  # index within the grid of predictors
        ind_y = int((true_arr[an,1] -  cy_min) / ybinsize)

        ind_x = min(  max(ind_x, 0), pred_shape[0]-1)
        ind_y = min(  max(ind_y, 0), pred_shape[1]-1)
        # Extended diagnostic in case something's off
        '''
        if False:  # only enabled for diagnostics
            if not (assigned_counts[ind_x, ind_y] < pred_shape[2]):  # Extended diagnostic.
                print("true_to_pred_grid: Error: Have already added ",assigned_counts[ind_x, ind_y]," out of a maximum of ",gridYi.shape[2],
                "possible 'slots' to predictive-grid cell [",ind_x,",",ind_y,"].  Increase last dimstion of pred_shape.")
                print("     img_filename = ",img_filename)
                print("     true_arr = ",true_arr)
                print("     an = ",an,", true_arr[an] = ",true_arr[an])
                print("     gridYi[",ind_x,",",ind_y,"] = ",gridYi[ind_x,ind_y])
                print("     xbinsize, ybinsize = ",xbinsize, ybinsize)

                err_img_filename = "true_to_pred_grid_err.png"
                print("     Drawing to ",err_img_filename)
                img = load_img(img_filename)                 # this is a PIL image
                #img_dims = img_to_array(img).shape
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)      # convert from PIL to OpenCV

                for an2 in range(true_arr.shape[0]):
                    print("     Drawing ellipse ",an2," of ",true_arr.shape[0],": ",true_arr[an2])
                    [cx, cy, a, b, cos2t, sin2t, noobj, rings] = true_arr[an2]   # ellipse for Testing
                    cx, cy,  a, b = int(round(cx)), int(round(cy)),  int(round(a)), int(round(b))  # OpenCV wants ints or it barfs
                    angle = np.rad2deg( np.arctan2(sin2t,cos2t)/2.0 )
                    draw_ellipse(img, [cx, cy], [a,b], angle, color=red, thickness=2)
                cv2.imwrite(err_img_filename,img)
            '''
        assert( assigned_counts[ind_x, ind_y] < pred_shape[2] )  # make sure we're not exceeding our allotment of predictors per grid location

        gridYi[ind_x,ind_y, assigned_counts[ind_x, ind_y]] = true_arr[an]
        assigned_counts[ind_x, ind_y] = assigned_counts[ind_x, ind_y] + 1
    return gridYi


def add_to_stack(a,b):
    """
    makes a list of lists of lists. Indices are  [image i][antinode j][params within antinode j]
    """
    if (a is None):
        return [b]
    return a + [b]


def nearest_multiple( a, b ):   # returns number smaller than a, which is the nearest multiple of b
    return  int(a/b) * b


def parse_meta_file(meta_filename):
    col_names = ['cx', 'cy', 'a', 'b', 'angle', 'rings']
    df = pd.read_csv(meta_filename,header=None,names=col_names)  # read metadata file
    df.drop_duplicates(inplace=True)  # sometimes the data from Zooniverse has duplicate rows

    arrs = []    # this is a list of lists, containing all the ellipse info & ring counts for an image
    for index, row in df.iterrows() :
        cx, cy = row['cx'], row['cy']
        a, b = row['a'], row['b']
        angle, num_rings = float(row['angle']), row['rings']

        # For definiteness: make sure a > b; but if we swap them, then adjust the angle
        if (b > a):
            a, b = b, a            # swap
            angle = angle + 90     # could subtract 90 instead; we're going to just take sin & cos of 2*angle later anyway

        # Input format (from file) is [cx, cy,  a, b, angle, num_rings]
        #    But we'll change that to [cx, cy, a, b, cos(2*angle), sin(2*angle), 0 (noobj=0, i.e. existence), num_rings] for ease of transition to classification
        if (num_rings > 0.0):    # Actually, check for existence
            tmp_arr = [cx, cy, a, b, np.cos(2*np.deg2rad(angle)), np.sin(2*np.deg2rad(angle)), 0, num_rings]
            arrs.append(tmp_arr)
        else:
            pass  # do nothing.  default is no ellipses in image

    arrs = sorted(arrs,key=itemgetter(0,1))     # sort by y first, then by x

    return arrs


def build_Y(total_load, meta_file_list, img_file_list, pred_grid=[6,6,2], set_means_ranges=False):
    """
    Reads in metadata and assigns them to the target output "Y" that the network is to be trained on

    """

    pred_shape = [pred_grid[0],pred_grid[1],pred_grid[2],cf.vars_per_pred]  # shape of output predictions = grid_shape * vars per_grid
    pred_shape = np.array(pred_shape,dtype=np.int)
    num_outputs = np.prod(np.array(pred_shape))

    true_stack = None                            # array stack to hold true info, to convert into Y
    for i in range(total_load):                 # read all true info from disk into arrays
        img_filename = img_file_list[i]
        meta_filename = meta_file_list[i]
        if (0 == i % 5000):
            print("      Reading metadata file i =",i,"/",total_load,":",meta_filename)
        one_true_arr = np.array(parse_meta_file(meta_filename)).tolist()     # one_true_arr is a list of the true info on all antinodes in this particular file
        true_stack = add_to_stack(true_stack, one_true_arr)  # add to the stack, to further parse later

    print("          Using annotations from metadata to setup 'true answers' Y and grid of predictors...")
    true_stack = np.array(true_stack)

    Y = np.zeros([total_load,num_outputs],dtype=cf.dtype)          # allocate Ytrue
    for i in range(total_load):                         # Divvy up all true values to grid of predictors
        # add true values to Y according to which 'grid cell' they apply to
        gridYi = true_to_pred_grid(np.array(true_stack[i]), pred_shape, img_filename=img_file_list[i])
        Y[i,:] = gridYi.flatten()      # Keras wants our output Y to be flat

    # Now that Y is fully read-in and flattened, do some operations on it...
    Y = norm_Y(Y, set_means_ranges=set_means_ranges)  # after all parts of Y are assigned, normalize

    return Y, pred_shape              # pred_shape tells how to un-flatten Y


## New: for parallel read of images into "X" array
mp_shared_array = None                               # global variable for array
def load_X_one_proc(img_file_list, force_dim, grayscale, i):
    global mp_shared_array

    # tmp will end up pointing to the memory address of the shared array we want to populate
    tmp = np.ctypeslib.as_array(mp_shared_array)

    img_filename = img_file_list[i]
    if (0 == i % 2000):
        print("      Reading image file i =",i,"/",len(img_file_list),":",img_filename)

    img = load_img(img_filename)
    if (force_dim is not None):         # resize image if needed
        img = img.resize((force_dim,force_dim), PIL.Image.ANTIALIAS)
    img = img_to_array(img)

    img = img/255.0  # scale from 0 to 1
    img -= 0.5       # zero-mean;  for Inception or Xception
    img *= 2.        # for Inception or Xception

    # copy from the individual image into the dataset
    if (grayscale):
        tmp[i,:,:,0] = img[:,:,0]   # let's throw out the RGB and just keep greyscale
    else:
        tmp[i,:,:,:] = img[:,:,:]   # (default) keep RGB, even though ESPI images are greyscale, b/c many CNN models expect RGB
    return


def build_X(total_load, img_file_list, force_dim=224, grayscale=False):
    """
    Reads in images and assigns them to the input "X" of the network
    """
    global mp_shared_array
    mp_shared_array = None

    print("      Reading images and assigning as input X...")
    img_filename = img_file_list[0]
    print("          First image file = ",img_filename)
    img = load_img(img_filename, grayscale=grayscale)  # this is a PIL image

    if (force_dim is not None):                 # resize if needed (to square image)
        print(f"          Resizing images to {force_dim}x{force_dim}")
        img = img.resize((force_dim,force_dim), PIL.Image.ANTIALIAS)

    img = img_to_array(img)
    img_dims = img.shape
    print("          img_dims = ",img_dims)
    # Allocate storage
    if (grayscale):
        X = np.zeros((total_load, img_dims[0], img_dims[1],1),dtype=cf.dtype)
    else:
        X = np.zeros((total_load, img_dims[0], img_dims[1], img_dims[2]),dtype=cf.dtype)

    # Fill X array by reading image files (in parallel)
    #      Next line requires numpy>=1.17 or else very large arrays will give an error
    tmp = np.ctypeslib.as_ctypes(X)             # tmp variable avoids numpy garbage-collection bug
    print("          Allocating shared storage for multiprocessing (this may take a while)")
    mp_shared_array = sharedctypes.RawArray(tmp._type_, tmp)
    num_procs = cpu_count()                     # set num_procs = 1 to recover serial execution
    print("          Parallelizing image reads across",num_procs,"processes.  (Numbers may appear out of order.)")
    #with get_context("spawn").Pool(num_procs) as p:
    p = Pool(num_procs)
    wrapper = partial(load_X_one_proc, img_file_list[0:total_load], force_dim, grayscale)
    result = p.map(wrapper, range(total_load))                # here's where we farm out the op
    X = np.ctypeslib.as_array(mp_shared_array, shape=X.shape)  # this actually happens pretty fast
    p.close()
    p.join()

    # Next couple lines are here just to force garbage collection
    mp_shared_array, tmp = None, None
    gc.collect()

    # parallelized, as above
    """
    # TODO: parallelize this?
    for i in range(total_load):     # image info into X array
        # TODO: make this parallel, e.g. using Multiprocessing
        img_filename = img_file_list[i]
        if (0 == i % 1000):
            print("      Reading image file i =",i,"/",total_load,":",img_filename)

        img = load_img(img_filename)
        if (force_dim is not None):         # resize image if needed
            img = img.resize((force_dim,force_dim), PIL.Image.ANTIALIAS)
        img = img_to_array(img)

        img = img/255.0  # scale from 0 to 1
        img -= 0.5       # zero-mean;  for Inception or Xception
        img *= 2.        # for Inception or Xception

        # copy from the individual image into the dataset
        if (grayscale):
            X[i,:,:,0] = img[:,:,0]   # let's throw out the RGB and just keep greyscale
        else:
            X[i,:,:,:] = img[:,:,:]   # (default) keep RGB, even though ESPI images are greyscale, b/c many CNN models expect RGB
    """

    return X, img_dims



def build_dataset(path="Train/", load_frac=1.0, set_means_ranges=False,
        pred_grid=[6,6,2], batch_size=None, shuffle=True):
    """
    builds the Training or Test data set
    Inputs:
      path              Pathname (including /) to read image and metadata files from
      load_frac         Fraction of total amount of data to read from (useful to decrease dataset size for debugging)
      set_means_ranges  Sets the values of the mean of the data and the range. False=use existing values
                            For training set, use True, for others use False
      #grayscale         Use grascale images or not.  Set to False as most pre-made CNNs require 3 color slots
      #force_dim         Resize images to a square of this size. Again, for pre-made CNN models
      pred_grid         How big the output grid of predictors should be: a 6x6 grid with 2 predictors per 'cell' is useful
      batch_size        Used only to force the dataset size to be a multiple of this number
    Outputs:
      X, Y              input and target data
      pred_shape        Tells how to unflatten Y meaningfully. (it's equal to pred_grid.shape with cf.vars_per_pred tacked on the end)
    """
    global means, ranges

    if cf.model_type == 'simple':
        grayscale = False
        force_dim = 224
    else:
        grayscale = True
        force_dim = 331

    print("Loading data from",path,", fraction =",load_frac)
    #-------------------------------------------
    # Setup: Get lists of files to read from
    #-------------------------------------------
    img_file_list = sorted(glob.glob(path+'*.png'))
    meta_file_list = sorted(glob.glob(path+'*'+cf.meta_extension))
    print("      Check: len(img_file_list) = "+str(len(img_file_list))+", and len(meta_file_list) = "+str(len(meta_file_list)) )
    assert len(img_file_list) == len(meta_file_list), "Error: len(img_file_list) = " \
        +str(len(img_file_list))+" but len(meta_file_list) = "+str(len(meta_file_list))

    # Shuffle, images & metadata (together)
    #      https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
    if shuffle:
        c = list(zip(img_file_list, meta_file_list))
        random.shuffle(c)
        img_file_list, meta_file_list = zip(*c)

    # total loaded is a fraction of total files in directory
    total_files = len(img_file_list)
    total_load = int(total_files * load_frac)
    if (batch_size is not None):                # keras gets particular: dataset size must be mult. of batch_size
        total_load = nearest_multiple( total_load, batch_size)
    print("      Total files = ",total_files,", going to load total_load = ",total_load)

    #-------------------------------------------
    # Now that we have our list of images and list of metadata, we can build the dataset
    #-------------------------------------------
    Y, pred_shape = build_Y(total_load, meta_file_list, img_file_list, pred_grid=pred_grid, set_means_ranges=set_means_ranges)
    X, img_dims = build_X(total_load, img_file_list, force_dim=force_dim, grayscale=grayscale)

    # all data read in, so return
    return X, Y, img_file_list, pred_shape
