
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


dtype = np.float32

# Define some colors: openCV uses BGR instead of RGB
blue = (255,0,0)
red = (0,0,255)
green = (0,255,0)
white = (255)
black = (0)
grey = (128)

# define indices for different parts of the data stream
vars_per_pred = 7
ind_cx = 0
ind_cy = 1
ind_semi_a = 2
ind_semi_b = 3
ind_angle = 4
ind_noobj = 5
ind_rings = 6



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
def show_pred_ellipses(Yt, Yp, file_list, num_draw=30, log_dir='./logs/', ind_extra=None):
    # Yt & Yp are already de-normed.  Yt = true,  Yp= predict
    m = Yt.shape[0]
    num_draw = min(num_draw,m)

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

            [cx, cy, a, b, angle, noobj, rings] = Yt[j,an*vars_per_pred:(an+1)*vars_per_pred]   # ellipse for Testing
            cx, cy,  a, b, noobj, rings = int(round(cx)), int(round(cy)),  int(round(a)), int(round(b)), int(round(noobj)), int(round(rings)) # OpenCV wants ints or it barfs


            if (an==0):
                print("       True:   {: >5d}, {: >5d},  {: >3d},  {: >5d}, {: >5d},   {: >6.2f}".format(cx, cy, rings, a, b, angle))
            if (noobj < 0.5) and (rings > 0) and (a>=0) and (b>=0):  # only draw if you should and if you can
                draw_ellipse(img, [cx, cy], [a,b], angle, color=red, thickness=1)
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy+2), cv2.FONT_HERSHEY_TRIPLEX, 0.85, black, lineType=cv2.LINE_AA);  # add a little outline for readibility
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy), cv2.FONT_HERSHEY_TRIPLEX, 0.75, red, lineType=cv2.LINE_AA);

            [cx, cy, a, b, angle, noobj, rings] = Yp[j,an*vars_per_pred:(an+1)*vars_per_pred]   # ellipse for Prediction
            cx, cy,  a, b, noobj, rings = int(round(cx)), int(round(cy)),  int(round(a)), int(round(b)), int(round(noobj)), int(round(rings)) # OpenCV wants ints or it barfs

            if (an==0):
                print("       Pred:   {: >5d}, {: >5d},  {: >3d},  {: >5d}, {: >5d},   {: >6.2f}".format(cx, cy, rings, a, b, angle))
            if (noobj < 0.5) and (rings > 0) and (a>=0) and (b>=0):  # only draw if you should and if you can
                draw_ellipse(img, [cx, cy], [a,b], angle, color=green, thickness=1)
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy+27), cv2.FONT_HERSHEY_TRIPLEX, 0.85, black, lineType=cv2.LINE_AA);     # dark outline
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy+25), cv2.FONT_HERSHEY_TRIPLEX, 0.75, green, lineType=cv2.LINE_AA);

        cv2.imwrite(out_filename,img)           # save output image

    return

def iou_score(ell_1, ell_2):  # compute intersection-over-union for two rotated ellipses
    # for now: ingore rotation, and treat ellipses as boxes
    # TODO: come back and fix this
    return



def custom_loss_old(y_true, y_pred):
    # first sum up the squared error column-wise
    sqerr = K.square(y_true - y_pred)
    loss = K.sum(sqerr, axis=-1)

    # subtract the loss for the sliced part
    loss -= K.sum(sqerr[:, 4:-1:7], axis=-1)

    # add back the adjusted loss for the sliced part
    numerator = y_true[:, 2:-1:7] - y_true[:, 3:-1:7]
    loss += K.sum(sqerr[:, 4:-1:7] * K.square(numerator ), axis=-1)

    # take average
    ncols = K.int_shape(y_pred)[-1]
    loss /= ncols
    return K.mean(loss)



def my_loss_old(y_true, y_pred): # The following is only suitable for numpy arrays, not for Keras/TF/Theano tensors
    # This is MSE but the angle is term is specially weighted:
    #     multiply angle by a-b  (so angle matters less for circles)  use a & b from true
    sqerr = (y_true-y_pred)**2
    sqerr[:,ind_angle:-1:vars_per_pred] *= (y_true[:,ind_semi_a:-1:vars_per_pred] - y_true[:,ind_semi_b:-1:vars_per_pred])**2
    return sqerr.mean()


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


def true_to_pred_grid(true_arr, pred_shape, num_classes=11, img_filename=None):    # the essence of the YOLO-style approach
                                    # this takes our 'true' antinode info, and assigns it across the 'grid' of predictors, i.e. YOLO-style
                                    # true_arr is a list of antinode data which has been read from a text file
                                    # pred_shape has dimensions [nx, nx, preds_per_cell, vars_per_pred]
    # TODO: and each value is organized (according to loss type) as
    #   cx, cy, a, b, angle, prob_exist (or 0 rings), prob_1_ring, prob_2_rings,..., prob_11_rings
    #   ... 17 variables in all
    global means, ranges

    true_arr = np.array(true_arr, dtype=dtype)   # convert from list to array
    xbinsize = int(orig_img_dims[0] / pred_shape[0])
    ybinsize = int(orig_img_dims[1] / pred_shape[1])

    # Also, set up the means & ranges for normalization, according to the grid of predictors
    gridmeans = np.zeros(pred_shape,dtype=dtype)
    gridranges = np.zeros(pred_shape,dtype=dtype)
    griddefaults = np.zeros(pred_shape,dtype=dtype)
    for i in range(pred_shape[0]):
        for j in range(pred_shape[1]):
            grid_cx = i*xbinsize + xbinsize/2
            grid_cy = j*ybinsize + ybinsize/2
            #           format: [cx,         cy,           a,              b,        angle,  noobj,  num_rings]   noobj = 0/1 flag for background
            gridmeans[i,j] =    [grid_cx,    grid_cy,   xbinsize/2,    ybinsize/2,    90.0,   0.5,    5] # + [0]*(num_classes+1)
            gridranges[i,j] =   [xbinsize,  ybinsize,   xbinsize,      ybinsize,     180.0,     1,   10] #+ [1]*(num_classes+1)
            griddefaults[i,j] = [grid_cx,    grid_cy,   xbinsize/2,    ybinsize/2,    90.0,     1,    0] # default is noobj=1, rings=0

    gridYi = np.copy(griddefaults)              # initialize a single grid-Y output with default values

    means = gridmeans.flatten()                 # assign global means & ranges for later
    ranges = gridranges.flatten()

    # Now here's where we actually assign the true_arr values
    #print("divvy_up_true: true_arr = ",true_arr)
    assigned_counts = np.zeros(gridYi.shape[0:2],dtype=np.int)   # count up how many times a given array has been assigned
    for an in range(true_arr.shape[0]):
        #print("      true_arr[an,0] = ",true_arr[an,0],",  xbinsize, ybinsize = ",xbinsize, ybinsize)
        ind_x = int(true_arr[an,0] / xbinsize)  # index within the grid of predictors
        ind_y = int(true_arr[an,1] / ybinsize)
        #print("            ind_x, ind_y = ",ind_x, ind_y,", assigned_counts[ind_x, ind_y] =",assigned_counts[ind_x, ind_y])
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



# builds the Training or Test data set
def build_dataset(path="Train/", load_frac=1.0, set_means_ranges=False, grayscale=False, force_dim=224, pred_grid=[5,5,3]):
    global means, ranges

    img_file_list = sorted(glob.glob(path+'steelpan*.bmp'))
    txt_file_list = sorted(glob.glob(path+'steelpan*.txt'))
    assert( len(img_file_list) == len(txt_file_list))

    total_files = len(img_file_list)
    total_load = int(total_files * load_frac)
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

    for i in range(total_load):
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

        # Y holds  [ xc, yc, rings (1-11), a, b, theta (0-180)], multiple times
        true_arr = parse_txt_file(txt_filename)     # true_arr is a list of the true info on all antinodes in this file
        num_antinodes = int(round(len(true_arr)*1.0/vars_per_pred))              # number of antinodes in this particular file

        gridYi = true_to_pred_grid(true_arr, pred_shape, img_filename=img_filename)     # add true values to y according to which 'grid cell' they apply to

        # Keras wants our output Y to be flat
        Y[i,:] = gridYi.flatten()

        #if (i==0):
        #    print("       Y[0] = ",Y[0])

    Y = norm_Y(Y, set_means_ranges=set_means_ranges)  # after all parts of Y are assigned, normalize

    return X, Y, img_dims, img_file_list, pred_shape              # pred_shape tells how to un-flatten Y

lambda_center = 1.0
lambda_size = 1.0
lambda_angle = 30.0
lambda_noobj = 0.3
lambda_class = 10.0

def custom_loss(y_true, y_pred):  # it's just MSE but the angle term is weighted by (a-b)^2
    print("custom_loss function engaged!")
    sqerr = K.square(y_true - y_pred)   # loss is 'built on' squared error
    pobj = 1 - y_true[:, ind_noobj::vars_per_pred]   # probability of object", i.e. existence.  if no object, then we don't care about the rest of the variables

    loss = lambda_center * ( K.sum(pobj* sqerr[:,ind_cx::vars_per_pred],     axis=-1) + K.sum(pobj* sqerr[:,ind_cy:-1:vars_per_pred], axis=-1))
    loss += lambda_size  * ( K.sum(pobj* sqerr[:,ind_semi_a::vars_per_pred], axis=-1) + K.sum(pobj* sqerr[:,ind_semi_b:-1:vars_per_pred], axis=-1))
    abdiff = y_true[:, ind_semi_a::vars_per_pred] - y_true[:, ind_semi_b:-1:vars_per_pred]
    loss += lambda_angle * K.sum(pobj* sqerr[:, ind_angle::vars_per_pred] * K.square(abdiff) , axis=-1)
    loss += lambda_noobj * K.sum(sqerr[:, ind_noobj::vars_per_pred], axis=-1)
    loss += lambda_class * K.sum( pobj * sqerr[:, ind_rings::vars_per_pred], axis=-1)

    # take average
    ncols = K.int_shape(y_pred)[-1]
    loss /= ncols
    return K.mean(loss)


def my_loss(y_true, y_pred):  # same as custom_loss but via numpy for easier diagnostics; inputs should be numpy arrays, not Tensors
    y_shape = y_pred.shape
    print("    my_loss: y_shape = ",y_shape)
    sqerr = (y_true - y_pred)**2   # loss is 'built on' squared error
    pobj = 1 - y_true[:, ind_noobj::vars_per_pred]   # probability of object", i.e. existence.  if no object, then we don't care about the rest of the variables

    center_loss = lambda_center * (np.sum(pobj* sqerr[:,ind_cx::vars_per_pred],     axis=-1) + np.sum(pobj* sqerr[:,ind_cy::vars_per_pred], axis=-1))
    size_loss   = lambda_size  * ( np.sum(pobj* sqerr[:,ind_semi_a::vars_per_pred], axis=-1) + np.sum(pobj* sqerr[:,ind_semi_b::vars_per_pred], axis=-1))
    abdiff = y_true[:, ind_semi_a::vars_per_pred] - y_true[:, ind_semi_b::vars_per_pred]
    angle_loss  = lambda_angle * np.sum(pobj * sqerr[:, ind_angle::vars_per_pred] * (abdiff**2) , axis=-1)
    noobj_loss  = lambda_noobj * np.sum(sqerr[:, ind_noobj::vars_per_pred], axis=-1)
    class_loss  = lambda_class * np.sum(pobj * sqerr[:, ind_rings::vars_per_pred], axis=-1)

    losses = np.array([center_loss, size_loss, angle_loss, noobj_loss, class_loss])
    losses = np.mean(losses,axis=-1)
    big_ind = np.argmax(losses)

    print("    my_loss: by class: [   center,        size,          angle,           noobj,          class]")
    print("              losses =",losses,", ind of biggest =",big_ind)
    loss = np.sum(losses)
    # take average
    ncols = y_pred.shape[-1]
    loss /= ncols
    return loss



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

        # Input format (from file) is [cx, cy, num_rings, a, b, angle]
        #    But we'll change that to [cx, cy, a, b, angle, 0 (noobj=0, i.e. existence), num_rings] for ease of transition to classification
        tmp_arr = subarr[:]  # clone the list
        tmp_arr[2:5] = subarr[3:6]  # shift last three vars to the left
        tmp_arr[ind_noobj] = 0  # noobj = 0, i.e. object exists
        tmp_arr = tmp_arr + [subarr[2]] # move num_rings to end

        arrs.append(tmp_arr)

    arrs = sorted(arrs,key=itemgetter(0,1))     # sort by y first, then by x


    return arrs
