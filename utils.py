import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import PIL
from operator import itemgetter

# Define some colors: openCV uses BGR instead of RGB
blue = (255,0,0)
red = (0,0,255)
green = (0,255,0)
white = (255)
black = (0)
grey = (128)



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
def show_pred_ellipses(Yt, Yp, file_list, num_draw=30, log_dir='./logs/', vars_per_pred=6, ind_extra=None):
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

            [cx, cy, rings, a, b, angle] = Yt[j,an*vars_per_pred:(an+1)*vars_per_pred]   # ellipse for Testing
            cx, cy, rings, a, b = int(round(cx)), int(round(cy)), int(round(rings)), int(round(a)), int(round(b))


            if (an==0):
                print("       True:   {: >5d}, {: >5d},  {: >3d},  {: >5d}, {: >5d},   {: >6.2f}".format(cx, cy, rings, a, b, angle))
            if (rings > 0) and (a>=0) and (b>=0):
                draw_ellipse(img, [cx, cy], [a,b], angle, color=red, thickness=1)
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy+2), cv2.FONT_HERSHEY_TRIPLEX, 0.85, black, lineType=cv2.LINE_AA);  # add a little outline for readibility
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy), cv2.FONT_HERSHEY_TRIPLEX, 0.75, red, lineType=cv2.LINE_AA);

            [cx, cy, rings, a, b, angle] = Yp[j,an*vars_per_pred:(an+1)*vars_per_pred]   # ellipse for Prediction
            cx, cy, rings, a, b = int(round(cx)), int(round(cy)), int(round(rings)), int(round(a)), int(round(b))
            if (an==0):
                print("       Pred:   {: >5d}, {: >5d},  {: >3d},  {: >5d}, {: >5d},   {: >6.2f}".format(cx, cy, rings, a, b, angle))
            if (rings > 0) and (a>=0) and (b>=0):
                draw_ellipse(img, [cx, cy], [a,b], angle, color=green, thickness=1)
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy+27), cv2.FONT_HERSHEY_TRIPLEX, 0.85, black, lineType=cv2.LINE_AA);     # dark outline
                cv2.putText(img, "{: >2d}".format(rings), (cx-10,cy+25), cv2.FONT_HERSHEY_TRIPLEX, 0.75, green, lineType=cv2.LINE_AA);

        cv2.imwrite(out_filename,img)           # save output image

    return



def my_loss(Ypred, Ytrue, vars_per_pred=6):  # this is MSE but the angle is deprecated
    sqerr = (Ypred - Ytrue)**2

    # multiple angle by a-b  (so angle matters less for circles)  use a & b from true
    max_pred_antinodes = int(Ytrue.shape[1]/vars_per_pred)
    for an in range(max_pred_antinodes):
        i_start = an * vars_per_pred
        ind_a = i_start + 3
        ind_b = i_start + 4
        ind_angle = i_start + 5
        sqerr[:,ind_angle] = sqerr[:,ind_angle] * (Ytrue[:,ind_a] - Ytrue[:,ind_b])

    return sqerr.mean()


orig_img_dims=[512,384]
#means1 =  [ orig_img_dims[0]/2.0,  orig_img_dims[1]/2.0,    5.0,  orig_img_dims[0]/4.0,    orig_img_dims[1]/4,    90.0]
#ranges1 = [ orig_img_dims[0],      orig_img_dims[1],       10.0,  orig_img_dims[0]/2.0,  orig_img_dims[1]/2.0,   180.0]

#default0 =  [ means1[0], means1[1],   0.0,  means1[3], means1[4], means1[5]] # no-op for zero rings
means = []
ranges = []
def norm_Y(Y, set_means_ranges=False):  # not using this yet, but might be handy
    global means, ranges
    if (False):  # this doesn't work. TODO: fix it!
        means = np.mean(Y,axis=0)
        ranges = np.var(Y,axis=0)    # variance
    print("means = ",means)
    print("ranges = ",ranges)
    return (Y-means)/ranges #,  means, ranges

def denorm_Y(normY):
    global means, ranges
    return normY*ranges + means


# TODO: remove item from list once assigned
def true_to_pred_grid(true_arr, pred_shape, img_filename=None):    # the essence of the YOLO-style approach
                                    # this takes our 'true' antinode info, and assigns it across the 'grid' of predictors, i.e. YOLO-style
                                    # true_arr is a list of antinode data which has been read from a text file
                                    # pred_shape has dimensions [nx, nx, preds_per_cell, vars_per_pred]
    # TODO: and each value is organized (according to loss type) as
    #   cx, cy, prob_exist, prob_1_ring, prob_2_rings,...prob_11_rings,
    global means, ranges

    true_arr = np.array(true_arr)   # convert from list to array

    xbinsize = int(orig_img_dims[0] / pred_shape[0])
    ybinsize = int(orig_img_dims[1] / pred_shape[1])

    # also, set up the means & ranges for normalization, according to the grid of predictors
    gridmeans = np.zeros(pred_shape,dtype=np.float32)
    gridranges = np.zeros(pred_shape,dtype=np.float32)
    griddefaults = np.zeros(pred_shape,dtype=np.float32)
    for i in range(pred_shape[0]):
        for j in range(pred_shape[1]):
            grid_cx = i*xbinsize + xbinsize/2
            grid_cy = j*ybinsize + ybinsize/2
            gridmeans[i,j] =    [grid_cx,    grid_cy,    5.0,  xbinsize/2,    ybinsize/2,    90.0]
            gridranges[i,j] =   [xbinsize,  ybinsize,   10.0,    xbinsize,      ybinsize,   180.0]
            griddefaults[i,j] = [grid_cx,    grid_cy,    0.0,  xbinsize/2,    ybinsize/2,    90.0]

    gridYi = np.copy(griddefaults)              # initialize a single grid-Y output with default values

    means = gridmeans.flatten()                 # assign global means & ranges for later
    ranges = gridranges.flatten()


    #print("divvy_up_true: true_arr = ",true_arr)
    assigned_counts = np.zeros(gridYi.shape[0:2],dtype=np.int)   # count up how many times a given array has been assigned
    for an in range(true_arr.shape[0]):
        #print("      true_arr[an,0] = ",true_arr[an,0],",  xbinsize, ybinsize = ",xbinsize, ybinsize)
        ind_x = int(true_arr[an,0] / xbinsize)
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
def build_dataset(path="Train/", load_frac=1.0, set_means_ranges=False, grayscale=False, pred_grid=[5,5,3], vars_per_pred=6, force_dim=None):
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


    if (force_dim is not None):                 # resize if needed
        print("        Resizing to ",force_dim,"x",force_dim)
        img = img.resize((force_dim,force_dim), PIL.Image.ANTIALIAS)

    img = img_to_array(img)
    img_dims = img.shape
    print("       img_dims = ",img_dims)

    pred_shape = [pred_grid[0],pred_grid[1],pred_grid[2],vars_per_pred]  # shape of output predictions = grid_shape * vars per_grid

    if (grayscale):
        X = np.zeros((total_load, img_dims[0], img_dims[1],1),dtype=np.float32)
    else:
        X = np.zeros((total_load, img_dims[0], img_dims[1], img_dims[2]),dtype=np.float32)

    pred_shape = np.array(pred_shape,dtype=np.int)
    num_outputs = np.prod(np.array(pred_shape))
    #print("pred_shape, num_outputs = ",pred_shape,num_outputs)
    Y = np.zeros([total_load,num_outputs],dtype=np.float32)          # but the final Y has to be flat (thanks Keras), not a grid

    for i in range(total_load):
        img_filename = img_file_list[i]
        txt_filename = txt_file_list[i]
        #if (i == 10739):
        #    print(" i = ",i," img_filename = ",img_filename,", txt_filename = ",txt_filename)

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
        true_arr = parse_txt_file(txt_filename, vars_per_pred=vars_per_pred)     # true_arr is a list of the true info on all antinodes in this file
        num_antinodes = int(round(len(true_arr)*1.0/vars_per_pred))              # number of antinodes in this particular file

        gridYi = true_to_pred_grid(true_arr, pred_shape, img_filename=img_filename)     # add true values to y according to which 'grid cell' they apply to

        # Keras wants our output Y to be flat
        Y[i,:] = gridYi.flatten()

        if (i==0):
            print("       Y[0] = ",Y[0])

    Y = norm_Y(Y, set_means_ranges=set_means_ranges)  # after all parts of Y are assigned, normalize

    return X, Y, img_dims, img_file_list, pred_shape              # pred_shape tells how to un-flatten Y


def parse_txt_file(txt_filename, vars_per_pred=6):
    f = open(txt_filename, "r")
    lines = f.readlines()
    f.close()

    xmin = 99999
    return_arr = np.zeros(vars_per_pred,dtype=np.float32).flatten()
    arrs = []
    for j in range(len(lines)):
        line = lines[j]   # grab a line
        line = line.translate({ord(c): None for c in '[] '})     # strip unwanted chars in unicode line
        string_vars = line.split(sep=',')

        vals = [float(numeric_string) for numeric_string in string_vars]

        arrs.append(vals[0:vars_per_pred])

    arrs = sorted(arrs,key=itemgetter(0,1))     # sort by y first, then by x
    #print("parse_txt_file: arrs = ",arrs)
    #arrs = np.array(arrs,dtype=np.float32).flatten()

    return arrs
