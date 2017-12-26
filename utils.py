
import numpy as np
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


dtype = np.float32

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
    num_draw = min(num_draw,m)

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
            cx, cy,  a, b, noobj, rings = int(round(cx)), int(round(cy)),  int(round(a)), int(round(b)), int(round(noobj)), int(round(rings)) # OpenCV wants ints or it barfs
            angle = np.rad2deg( np.arctan2(sin2t,cos2t)/2.0 )
            if (angle < 0):
                angle += 180

            if ((an < 6) and (noobj==0)):
                print("       True:   {: >4d}, {: >4d},  {: >4d}, {: >4d},   {: >6.2f},  {: >2d}, {: >4d}".format(cx, cy, a, b, angle, noobj, rings))
            if (noobj < 0.5) and (rings > 0) and (a>=0) and (b>=0):  # only draw if you should and if you can
                draw_ellipse(img, [cx, cy], [a,b], angle, color=red, thickness=2)
                #cv2.putText(img, "{: >2d}".format(rings), (cx-13,cy), cv2.FONT_HERSHEY_TRIPLEX, 0.95, grey, lineType=cv2.LINE_AA);  # add a little outline for readibility
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy+2), cv2.FONT_HERSHEY_TRIPLEX, 0.95, black, lineType=cv2.LINE_AA)  # add a little outline for readibility
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy), cv2.FONT_HERSHEY_TRIPLEX, 0.85, red, lineType=cv2.LINE_AA)

            [cx, cy, a, b, cos2t, sin2t, noobj, rings] = Yp[j,an*vars_per_pred:(an+1)*vars_per_pred]   # ellipse for Prediction
            cx, cy,  a, b, noobj, rings = int(round(cx)), int(round(cy)),  int(round(a)), int(round(b)), int(round(noobj)), int(round(rings)) # OpenCV wants ints or it barfs
            angle = np.rad2deg( np.arctan2(sin2t,cos2t)/2.0 )
            if (angle < 0):
                angle += 180

            if ((an < 6) and (noobj==0)):
                print("       Pred:   {: >4d}, {: >4d},  {: >4d}, {: >4d},   {: >6.2f},  {: >2d}, {: >4d}".format(cx, cy, a, b, angle, noobj, rings))
            if (noobj < 0.5) and (rings > 0) and (a>=0) and (b>=0):  # only draw if you should and if you can
                draw_ellipse(img, [cx, cy], [a,b], angle, color=green, thickness=2)
                #cv2.putText(img, "{: >2d}".format(rings), (cx-13,cy+27), cv2.FONT_HERSHEY_TRIPLEX, 0.95, grey, lineType=cv2.LINE_AA)     # white outline
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy+29), cv2.FONT_HERSHEY_TRIPLEX, 0.95, black, lineType=cv2.LINE_AA)     # dark outline
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy+27), cv2.FONT_HERSHEY_TRIPLEX, 0.85, green, lineType=cv2.LINE_AA)

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


def one_hot_list(target_ind, num):  # makes an list num elements long, of zeros, except at target_ind where it's 1
    one_hot = [0]*num
    one_hot[int(target_ind)] = 1
    return one_hot

def eval_classifier(one_hot_pred):   # one_hot_pred includes [ confidence of 0 rings / non-existence, confidence of 1 ring, ..., confidence of 11 rings ]
    # note that 'confidence' is similar to probability but is not necessarily bounded by 0,1
    # And note that YOLO authors don't use softmax activation.
    # So whichever confidence is highest, simply wins
    '''# if one_hot_pred[0] < 0.5:    # ok, this assumes we have a mean of zero between
    # return 0    # nothing exists here, return zero (as in zero rings)
    # ring_count = 1 + np.argmax(one_hot_pred[1:])     # don't include the 0-ring part when computing argmax'''
    # So all we're left with, then, is...
    return np.argmax(one_hot_pred)   # returns an integer, where 0 denotes nothing there

def set_Y_norms(Y, pred_shape):
    return


def true_to_pred_grid(true_arr, pred_shape, num_classes=11, img_filename=None):    # the essence of the YOLO-style approach
                                    # this takes our 'true' antinode info for one image, and assigns it across the 'grid' of predictors, i.e. YOLO-style
                                    # true_arr is a list of antinode data which has been read from a text file
                                    # pred_shape has dimensions [nx, nx, preds_per_cell, vars_per_pred]
    # TODO: and each value is organized (according to loss type) as
    #   cx, cy, a, b, angle, prob_exist (or 0 rings), prob_1_ring, prob_2_rings,..., prob_11_rings
    #   ... 17 variables in all
    global means, ranges

    #true_arr = np.array(true_arr, dtype=dtype)   # convert from list to array
    #[cx_min, cy_min] = [0, 0]
    #[cx_max, cy_max] = orig_img_dims
    [cx_min, cy_min] =  [40,  40]
    [cx_max, cy_max] =[ 470, 350]
    xbinsize = int( (cx_max - cx_min) / pred_shape[0])
    ybinsize = int( (cy_max - cy_min) / pred_shape[1])

    #xbinsize = int(orig_img_dims[0] / pred_shape[0])
    #ybinsize = int(orig_img_dims[1] / pred_shape[1])
    #cx_min = xbinsize / 2
    #cy_min = ybinsize / 2

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

    # Now here's where we actually assign the true_arr values
    #print("divvy_up_true: true_arr = ",true_arr)
    assigned_counts = np.zeros(gridYi.shape[0:2],dtype=np.int)   # count up how many times a given array has been assigned
    this_shape = true_arr.shape
    #print("   this_shape = ",this_shape)
    #print("   true_arr = ",true_arr)
    for an in range(true_arr.shape[0]):
        #print("      true_arr[an,0:2] = ",true_arr[an,0:2],",  xbinsize, ybinsize = ",xbinsize, ybinsize,", pred_shape = ",pred_shape)
        ind_x = int((true_arr[an,0]  - cx_min) / xbinsize)  # index within the grid of predictors
        ind_y = int((true_arr[an,1] -  cy_min) / ybinsize)
        #print("            ind_x, ind_y = ",ind_x, ind_y)

        ind_x = min(  max(ind_x, 0), pred_shape[0]-1)
        ind_y = min(  max(ind_y, 0), pred_shape[1]-1)
        #print("            ind_x, ind_y = ",ind_x, ind_y)
        #print("            assigned_counts[ind_x, ind_y] =",assigned_counts[ind_x, ind_y])
        if not (assigned_counts[ind_x, ind_y] < pred_shape[2]):
            print("true_to_pred_grid: Error: Have already added ",assigned_counts[ind_x, ind_y]," out of a maximum of ",gridYi.shape[2],
            "possible 'slots' to predictive-grid cell [",ind_x,",",ind_y,"].  Increase last dimstion of pred_shape.")
            print("     img_filename = ",img_filename)
            print("     true_arr = ",true_arr)
            print("     an = ",an,", true_arr[an] = ",true_arr[an])
            print("     gridYi[",ind_x,",",ind_y,"] = ",gridYi[ind_x,ind_y])
            print("     xbinsize, ybinsize = ",xbinsize, ybinsize)
            print("")
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

    img_file_list = sorted(glob.glob(path+'*.png'))
    txt_file_list = sorted(glob.glob(path+'*.txt'))
    assert( len(img_file_list) == len(txt_file_list))

    total_files = len(img_file_list)
    total_load = int(total_files * load_frac)
    if (batch_size is not None):                # keras gets particular: dataset size must be mult. of batch_size
        total_load = nearest_multiple( total_load, batch_size)

    print("       total files = ",total_files,", going to load total_load = ",total_load)

    i = 0
    img_filename = img_file_list[i]
    print("       first file = ",img_filename)
    img = load_img(img_filename)  # this is a PIL image

    print("       force_dim = ",force_dim)

    if (force_dim is not None):                 # resize if needed
        print("        Resizing to ",force_dim,"x",force_dim)
        img = img.resize((force_dim,force_dim), PIL.Image.ANTIALIAS)

    img = img_to_array(img)
    img_dims = img.shape
    print("       img_dims = ",img_dims)

    pred_shape = [pred_grid[0],pred_grid[1],pred_grid[2],vars_per_pred]  # shape of output predictions = grid_shape * vars per_grid

    if (grayscale):
        X = np.zeros((total_load, img_dims[0], img_dims[1],1),dtype=dtype)
    else:
        X = np.zeros((total_load, img_dims[0], img_dims[1], img_dims[2]),dtype=dtype)

    pred_shape = np.array(pred_shape,dtype=np.int)
    num_outputs = np.prod(np.array(pred_shape))
    #print("pred_shape, num_outputs = ",pred_shape,num_outputs)
    Y = np.zeros([total_load,num_outputs],dtype=dtype)          # but the final Y has to be flat (thanks Keras), not a grid

    true_stack = None                            # array stack to hold true info, to convert into Y
    for i in range(total_load):                 # read all true info from disk into arrays
        img_filename = img_file_list[i]
        txt_filename = txt_file_list[i]
        #if (i == 10739):
        if (0 == i % 1000):
            print(" Reading in i =",i,"/",total_load,": img_filename =",img_filename,", txt_filename =",txt_filename)

        img = load_img(img_filename)
        if (force_dim is not None):         # resize image if needed
            img = img.resize((force_dim,force_dim), PIL.Image.ANTIALIAS)
        img = img_to_array(img)

        img = img/255.0  # scale from 0 to 1
        img -= 0.5 # zero-mean;  for Inception or Xception
        img *= 2.  # for Inception or Xception

        if (grayscale):
            X[i,:,:,0] = img[:,:,0]   # let's throw out the rgb-ness
        else:
            X[i,:,:,:] = img[:,:,:]    # throw out the rgb and just keep greyscale

        # one_true_arr holds  [ xc, yc, rings (1-11), a, b, theta (0-180)] for multiple antinodes, for one image
        one_true_arr = np.array(parse_txt_file(txt_filename)).tolist()     # one_true_arr is a list of the true info on all antinodes in this particular file
        #print("    txt_filename = ",txt_filename,", one_true_arr = ",one_true_arr)
        # add to the stack
        true_stack = add_to_stack(true_stack, one_true_arr)

    # ------- All data has been read from disk at this point ------
    true_stack = np.array(true_stack)

    if (set_means_ranges):    # do some analysis on the dataset, e.g. set predictor default locations on grid
        #cx_min, cx_max = np.min(true_stack[:,ind_cx::vars_per_pred]), np.max(true_stack[:,ind_cx::vars_per_pred])
        #cy_min, cy_max = np.max(true_stack[:,ind_cy::vars_per_pred]), np.max(true_stack[:,ind_cy::vars_per_pred])
        pass

    for i in range(total_load):                         # Divvy up all true values to grid of predictors
        gridYi = true_to_pred_grid(np.array(true_stack[i]), pred_shape, img_filename=img_file_list[i])     # add true values to y according to which 'grid cell' they apply to

        # Keras wants our output Y to be flat
        Y[i,:] = gridYi.flatten()

    # Now that Y is fully read-in and flattened, do some operations on it...

    Y = norm_Y(Y, set_means_ranges=set_means_ranges)  # after all parts of Y are assigned, normalize

    return X, Y, img_dims, img_file_list, pred_shape              # pred_shape tells how to un-flatten Y



def parse_txt_file(txt_filename):
    f = open(txt_filename, "r")
    lines = f.readlines()
    f.close()

    xmin = 99999
    arrs = []
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
