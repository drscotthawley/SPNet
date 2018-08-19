
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import PIL
from operator import itemgetter
import keras.backend as K
import tensorflow as tf
import os
import errno
import random

dtype = np.float32

meta_extension = ".csv"   # extension for metadata files

# Define some colors: openCV uses BGR instead of RGB
blue = (255,0,0)
red = (0,0,255)
green = (0,200,0)
white = (255, 255, 255)
black = (0, 0, 0)
grey = (128, 128, 128)

# define indices for different parts of the data stream
vars_per_pred = 8
ind_cx = 0
ind_cy = 1
ind_semi_a = 2
ind_semi_b = 3
ind_angle1 = 4          # cos(2*theta)
ind_angle2 = 5          # sin(2*theta)
ind_noobj = 6
ind_rings = 7


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
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)
    return ellipse


# draws a bunch of sample output files showing where ellipses are on images
def show_pred_ellipses(Yt, Yp, file_list, num_draw=40, log_dir='./logs/', ind_extra=None):
    # Yt & Yp are already de-normed.  Yt = true,  Yp= predict
    m = Yt.shape[0]
    num_draw = min(num_draw, m, len(file_list)-1)

    print("  format for drawing: [cx,     cy,    rings,     a,     b,      angle]")
    j = 0
    for count in range(num_draw+1):
        j = j + 1
        if (count >= num_draw):
            j = ind_extra          # extra plot that we specify, e.g. the worst prediction so far
        in_filename = file_list[j]

        out_filename = log_dir+'/steelpan_pred_'+str(j).zfill(5)+'.png'
        print("    Drawing on image",count,"of",num_draw,":",in_filename,", writing to",out_filename)
        img = load_img(in_filename)                 # this is a PIL image
        img_dims = img_to_array(img).shape

        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)      # convert from PIL to OpenCV
        img = opencvImage

        max_pred_antinodes = int(Yt[0].size / vars_per_pred)
        for an in range(max_pred_antinodes):    # count through all antinodes

            [cx, cy, a, b, cos2t, sin2t, noobj, rings] = Yt[j,an*vars_per_pred:(an+1)*vars_per_pred]   # ellipse for Testing
            cx, cy,  a, b, noobj = int(round(cx)), int(round(cy)),  int(round(a)), int(round(b)), int(round(noobj)) # OpenCV wants ints or it barfs
            angle = np.rad2deg( np.arctan2(sin2t,cos2t)/2.0 )
            if (angle < 0):
                angle += 180

            if (an < 6) and (0 == noobj):  # noobj==0 means there is an object
                print("       True:   {: >4d}, {: >4d},  {: >4d}, {: >4d},   {: >6.2f},  {: >2d}, {: >4.1f}".format(cx, cy, a, b, angle, noobj, rings))
            if (noobj < 0.5) and (rings > 0) and (a>=0) and (b>=0):  # only draw if you should and if you can
                draw_ellipse(img, [cx, cy], [a,b], angle, color=red, thickness=2)
                #cv2.putText(img, "{: >2d}".format(rings), (cx-13,cy), cv2.FONT_HERSHEY_TRIPLEX, 0.95, grey, lineType=cv2.LINE_AA);  # add a little outline for readibility
                cv2.putText(img, "{: >3.1f}".format(rings), (cx-10,cy+2), cv2.FONT_HERSHEY_TRIPLEX, 0.95, black, lineType=cv2.LINE_AA)  # add a little outline for readibility
                cv2.putText(img, "{: >3.1f}".format(rings), (cx-10,cy), cv2.FONT_HERSHEY_TRIPLEX, 0.85, red, lineType=cv2.LINE_AA)

            [cx, cy, a, b, cos2t, sin2t, noobj, rings] = Yp[j,an*vars_per_pred:(an+1)*vars_per_pred]   # ellipse for Prediction
            cx, cy,  a, b, noobj = int(round(cx)), int(round(cy)),  int(round(a)), int(round(b)), int(round(noobj)) # OpenCV wants ints or it barfs
            angle = np.rad2deg( np.arctan2(sin2t,cos2t)/2.0 )
            if (angle < 0):
                angle += 180

            if (an < 6) and (0 == noobj):
                print("       Pred:   {: >4d}, {: >4d},  {: >4d}, {: >4d},   {: >6.2f},  {: >2d}, {: >4.1f}".format(cx, cy, a, b, angle, noobj, rings))
            if (noobj < 0.5) and (rings > 0) and (a>=0) and (b>=0):  # only draw if you should and if you can
                draw_ellipse(img, [cx, cy], [a,b], angle, color=green, thickness=2)
                #cv2.putText(img, "{: >2d}".format(rings), (cx-13,cy+27), cv2.FONT_HERSHEY_TRIPLEX, 0.95, grey, lineType=cv2.LINE_AA)     # white outline
                cv2.putText(img, "{: >3.1f}".format(rings), (cx-10,cy+29), cv2.FONT_HERSHEY_TRIPLEX, 0.95, black, lineType=cv2.LINE_AA)     # dark outline
                cv2.putText(img, "{: >3.1f}".format(rings), (cx-10,cy+27), cv2.FONT_HERSHEY_TRIPLEX, 0.85, green, lineType=cv2.LINE_AA)

        cv2.putText(img, in_filename, (7, orig_img_dims[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, black, lineType=cv2.LINE_AA); # display the filename
        cv2.putText(img, in_filename, (5, orig_img_dims[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, lineType=cv2.LINE_AA); # display the filename

        cv2.imwrite(out_filename,img)           # save output image

    return

def iou_score(ell_1, ell_2):  # compute intersection-over-union for two rotated ellipses
    # for now: ingore rotation, and treat ellipses as boxes
    # TODO: come back and fix this
    return


orig_img_dims=[512,384]
means = []
ranges = []
def norm_Y(Y, set_means_ranges=False):  # not using this yet, but might be handy
    global means, ranges
    if (False):  # this doesn't work. TODO: fix it!
        means = np.mean(Y,axis=0)
        ranges = np.var(Y,axis=0)    # variance
    #print("means = ",means)
    #print("ranges = ",ranges)
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
       cx, cy, a, b, angle, prob_exist (or 0 rings), prob_1_ring, prob_2_rings,..., prob_11_rings
       ... 17 variables in all
    """
    global means, ranges

    # Assign Defaults ---------------------------------
    [cx_min, cy_min] =  [40,  40]
    [cx_max, cy_max] =[ 470, 350]
    xbinsize = int( (cx_max - cx_min) / pred_shape[0])
    ybinsize = int( (cy_max - cy_min) / pred_shape[1])

    # Also, set up the means & ranges for normalization, according to the grid of predictors
    gridmeans = np.zeros(pred_shape,dtype=dtype)
    gridranges = np.zeros(pred_shape,dtype=dtype)
    griddefaults = np.zeros(pred_shape,dtype=dtype)
    for i in range(pred_shape[0]):
        for j in range(pred_shape[1]):
            grid_cx = i*xbinsize + cx_min + xbinsize/2
            grid_cy = j*ybinsize + cy_min + ybinsize/2
            #           format: [cx,         cy,           a,              b,     cos2theta, sin2theta,  noobj,  num_rings]   noobj = 0/1 flag for background
            griddefaults[i,j] = [grid_cx,    grid_cy,   xbinsize/2,    ybinsize/2,      -1,      0,         1,    0] # default for 'blanks'.  So noobj=1 (nothing there),  rings=0, angle is 90 degrees
            gridmeans[i,j] =    [grid_cx,    grid_cy,   xbinsize/2,    ybinsize/2,       0,      0,       0.5,    5] # + [0]*(num_classes+1)
            gridranges[i,j] =   [xbinsize,  ybinsize,   xbinsize,      ybinsize,         2,      2,         1,   10] #+ [1]*(num_classes+1),  -1 to 1 is range of 2

    gridYi = np.copy(griddefaults)              # initialize a single grid-Y output with default values

    means = gridmeans.flatten()                 # assign global means & ranges for later
    ranges = gridranges.flatten()

    # Assign True Values -----------------------------
    assigned_counts = np.zeros(gridYi.shape[0:2],dtype=np.int)   # count up how many times a given array has been assigned
    this_shape = true_arr.shape
    for an in range(true_arr.shape[0]):
        ind_x = int((true_arr[an,0]  - cx_min) / xbinsize)  # index within the grid of predictors
        ind_y = int((true_arr[an,1] -  cy_min) / ybinsize)

        ind_x = min(  max(ind_x, 0), pred_shape[0]-1)
        ind_y = min(  max(ind_y, 0), pred_shape[1]-1)
        if not (assigned_counts[ind_x, ind_y] < pred_shape[2]):
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

        assert( assigned_counts[ind_x, ind_y] < pred_shape[2] )

        gridYi[ind_x,ind_y, assigned_counts[ind_x, ind_y]] = true_arr[an]
        assigned_counts[ind_x, ind_y] = assigned_counts[ind_x, ind_y] + 1
    return gridYi


def add_to_stack(a,b):  # makes a list of lists of lists. Indices are  [image i][antinode j][params within antinode j]
    # TODO: I am not proud of this part of the code, and the ensuing np.array(list()).tolist() craziness elsewhere, But
    #         having something working ATM is sufficient.
    if (a is None):
        return [b]
    return a + [b]
    #return np.concatenate((a,b[None,:])) # does not work when images contain different #s of antinodes


def nearest_multiple( a, b ):   # returns number smaller than a, which is the nearest multiple of b
    return  int(a/b) * b

# builds the Training or Test data set
def build_dataset(path="Train/", load_frac=1.0, set_means_ranges=False, grayscale=False, force_dim=224, pred_grid=[6,6,2], batch_size=None):
    global means, ranges

    #---- Setup: list of files to read from
    img_file_list = sorted(glob.glob(path+'*.png'))
    meta_file_list = sorted(glob.glob(path+'*'+meta_extension))
    print("    Check: len(img_file_list) = "+str(len(img_file_list))+", and len(meta_file_list) = "+str(len(meta_file_list)) )
    assert len(img_file_list) == len(meta_file_list), "Error: len(img_file_list) = " \
        +str(len(img_file_list))+" but len(meta_file_list) = "+str(len(meta_file_list))

    # Shuffle, https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
    c = list(zip(img_file_list, meta_file_list))
    random.shuffle(c)
    img_file_list, meta_file_list = zip(*c)

    # total loaded is a fraction of total files in directory
    total_files = len(img_file_list)
    total_load = int(total_files * load_frac)
    if (batch_size is not None):                # keras gets particular: dataset size must be mult. of batch_size
        total_load = nearest_multiple( total_load, batch_size)
    print("    Total files = ",total_files,", going to load total_load = ",total_load)


    #---- Assign "true" Y values, via metadata files, first (to detect potential problems faster)
    pred_shape = [pred_grid[0],pred_grid[1],pred_grid[2],vars_per_pred]  # shape of output predictions = grid_shape * vars per_grid
    pred_shape = np.array(pred_shape,dtype=np.int)
    num_outputs = np.prod(np.array(pred_shape))

    print("    Reading metadata files...")
    true_stack = None                            # array stack to hold true info, to convert into Y
    for i in range(total_load):                 # read all true info from disk into arrays
        img_filename = img_file_list[i]
        meta_filename = meta_file_list[i]
        if (0 == i % 5000):
            print("      Reading metadata file i =",i,"/",total_load,":",meta_filename)
        one_true_arr = np.array(parse_meta_file(meta_filename)).tolist()     # one_true_arr is a list of the true info on all antinodes in this particular file
        true_stack = add_to_stack(true_stack, one_true_arr)  # add to the stack, to further parse later


    #--- Setup Y (true) data from metadata annotations
    print("    Using annotations from metadata to setup 'true answers' Y and grid of predictors...")
    true_stack = np.array(true_stack)

    Y = np.zeros([total_load,num_outputs],dtype=dtype)          # allocate Ytrue
    for i in range(total_load):                         # Divvy up all true values to grid of predictors
        gridYi = true_to_pred_grid(np.array(true_stack[i]), pred_shape, img_filename=img_file_list[i])     # add true values to y according to which 'grid cell' they apply to
        Y[i,:] = gridYi.flatten()      # Keras wants our output Y to be flat

    # Now that Y is fully read-in and flattened, do some operations on it...
    Y = norm_Y(Y, set_means_ranges=set_means_ranges)  # after all parts of Y are assigned, normalize


    #---- Read images and assign them as input X
    print("    Now reading images and assigning as input X...")
    img_filename = img_file_list[0]
    print("       first image file = ",img_filename)
    img = load_img(img_filename)  # this is a PIL image

    if (force_dim is not None):                 # resize if needed (to square image)
        print("        Resizing images to force_dim = ",force_dim,"x",force_dim)
        img = img.resize((force_dim,force_dim), PIL.Image.ANTIALIAS)

    img = img_to_array(img)
    img_dims = img.shape
    print("       img_dims = ",img_dims)
    if (grayscale):
        X = np.zeros((total_load, img_dims[0], img_dims[1],1),dtype=dtype)
    else:
        X = np.zeros((total_load, img_dims[0], img_dims[1], img_dims[2]),dtype=dtype)

    # TODO: parallelize this?
    for i in range(total_load):     # image info into X array
        # TODO: make this parallel, e.g. using Multiprocessing
        img_filename = img_file_list[i]
        if (0 == i % 1000):
            print("    Reading image file i =",i,"/",total_load,":",img_filename)

        img = load_img(img_filename)
        if (force_dim is not None):         # resize image if needed
            img = img.resize((force_dim,force_dim), PIL.Image.ANTIALIAS)
        img = img_to_array(img)

        img = img/255.0  # scale from 0 to 1
        img -= 0.5       # zero-mean;  for Inception or Xception
        img *= 2.        # for Inception or Xception

        if (grayscale):
            X[i,:,:,0] = img[:,:,0]   # let's throw out the RGB and just keep greyscale
        else:
            X[i,:,:,:] = img[:,:,:]   # (default) keep RGB, even though ESPI images are greyscale, b/c many CNN models expect RGB

    # all data read in, so return
    return X, Y, img_dims, img_file_list, pred_shape              # pred_shape tells how to un-flatten Y




def parse_meta_file(meta_filename):
    col_names = ['cx', 'cy', 'a', 'b', 'angle', 'rings']
    df = pd.read_csv(meta_filename,header=None,names=col_names)  # read metadata file
    df.drop_duplicates(inplace=True)  # sometimes the data from Zooniverse has duplicate rows

    arrs = []    # this is a list of lists, containing all the ellipse info & ring counts for an image
    for index, row in df.iterrows() :
        cx, cy = row['cx'], row['cy']
        a, b = row['a'], row['b']
        angle, num_rings = float(row['angle']), row['rings']
        # Input format (from file) is [cx, cy,  a, b, angle, num_rings]
        #    But we'll change that to [cx, cy, a, b, cos(2*angle), sin(2*angle), 0 (noobj=0, i.e. existence), num_rings] for ease of transition to classification
        if (num_rings > 0.0):    # Actually, check for existence
            tmp_arr = [cx, cy, a, b, np.cos(2*np.deg2rad(angle)), np.sin(2*np.deg2rad(angle)), 0, num_rings]
            arrs.append(tmp_arr)
        else:
            pass  # do nothing.  default is no ellipses in image

    arrs = sorted(arrs,key=itemgetter(0,1))     # sort by y first, then by x
    return arrs


'''
def old_parse_meta_file(meta_filename):  # old .txt format, no longer used
    f = open(meta_filename, "r")
    lines = f.readlines()
    f.close()

    arrs = []    # this is a list of lists, containing all the ellipse info & ring counts for an image
    for j in range(len(lines)):
        line = lines[j]   # grab a line
        line = line.translate({ord(c): None for c in '[] '})     # strip unwanted chars in unicode line
        string_vars = line.split(sep=',')
        vals = [float(numeric_string) for numeric_string in string_vars]
        subarr = vals[0:vars_per_pred]  # note vars_per_pred includes a slot for no-object, but numpy slicing convention means subarr will be vars_per_pred-1 elements long

        # Input format (from file) is [cx, cy,  a, b, angle, num_rings]
        #    But we'll change that to [cx, cy, a, b, cos(2*angle), sin(2*angle), 0 (noobj=0, i.e. existence), num_rings] for ease of transition to classification
        [cx, cy, a, b, angle, num_rings] = subarr
        tmp_arr = [cx, cy, a, b, np.cos(2*np.deg2rad(angle)), np.sin(2*np.deg2rad(angle)), 0, num_rings]
        #tmp_arr = subarr[:]  # clone the list
        #tmp_arr[2:5] = subarr[3:6]  # shift last three vars to the left
        #tmp_arr[ind_noobj] = 0  # noobj = 0, i.e. object exists
        #tmp_arr = tmp_arr + [subarr[2]] # move num_rings to end

        arrs.append(tmp_arr)

    arrs = sorted(arrs,key=itemgetter(0,1))     # sort by y first, then by x
    return arrs
'''
