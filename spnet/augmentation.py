# File to hold various data augmentation routines.
# Some of these are used "on the fly" by AugmentOnTheFly in callbacks.py
# Others are used by ../augment_preproc.py and ../gen_fake_espi.py

import numpy as np
import cv2
import random
import glob

def bandpass_mixup(
    img_in,  # the input image, to be augmented by mixing with a real image
    path_real='/home/shawley/datasets/parsed_zooniverze_steelpan/' # where the real images are stored
    ):
    '''
    For more realistic-looking images, replace low & high frequency components ('background')
    of fake images using those components from real images

    Note: assumes images are monochrome, i.e. no "channels" dimension
    '''

    # get a random background from the group of 'true' images
    file_true = random.choice(glob.glob(path_real+'/*.png'))
    img_true = cv2.imread(file_true, cv2.IMREAD_GRAYSCALE)
    # maybe flip the image
    flipchoice = np.random.choice([-1,0,1,2])
    if (flipchoice != 2):
        img_true = cv2.flip(img_true, flipchoice)

    # take fourier transforms of fake and true images
    dft_true = cv2.dft(np.float32(img_true), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_true = np.fft.fftshift(dft_true)    # center the "dc" part of image

    if len(img_in.shape) > 2:  # dft usage assumes greyscale
        dft_fake = cv2.dft(np.float32(cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)), flags=cv2.DFT_COMPLEX_OUTPUT)
    else:
        dft_fake = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_fake = np.fft.fftshift(dft_fake)    # center the "dc" part of image

    # Set up a filter: Keep the Lows and the Highs
    # create a rectagular mask first, center square is 1, remaining all zeros. LPF
    rows, cols = img_in.shape
    crow,ccol = rows//2 , cols//2
    wl, wh = 8, 0     #  width for LPF and HPF respectively. Chosen by experimentation
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-wl:crow+wl, ccol-wl:ccol+wl] = 1   # LPF
    if wh > 0:
        mask[0:wh,:] = 1    # HPF
        mask[-wh:,:] = 1    # HPF
        mask[:,0:wh] = 1    # HPF
        mask[:,-wh:] = 1    # HPF
    fshift = np.random.rand()*3*dft_shift_true*mask + (1-mask)*dft_shift_fake   # L/H from true, mids from fake

    # inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)

    if len(img_back.shape) != len(img_in.shape): #  in case we need to add channels
        img_back = cvCvtColor(img_back, output, CV_GRAY2BGR)

    return np.clip(img_back, 0, 255)



def blur_inplace(img, blur_prob=0.3, kernel_size=None):
    blur_dice_roll = np.random.random()
    if (blur_dice_roll <= blur_prob):
        ksize = kernel_size if kernel_size else random.choice([3,7])  # 7 would be slow
        cv2.GaussianBlur(img, (ksize,ksize), 0)



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




def cutout_inplace(img, max_regions=6, minsize=11, maxsize=75):
    """
    "Improved Regularization of Convolutional Neural Networks with Cutout", https://arxiv.org/abs/1708.04552
    All we do is chop out rectangular regions from the image, i.e. "masking out contiguous sections of the input"
    Unlike the original cutout paper, we don't cut out anything too huge, and we use random (greyscale) colors
    Note: operates IN PLACE on one 'image'.  image is really a numpy array
    """
    num_regions = np.random.randint(0, high=max_regions+1)
    if (0 == num_regions):
        return              # abort
    colormin, colormax = np.min(img), np.max(img)
    for region in range(num_regions):
        pt1 = ( np.random.randint(0,img.shape[0]-minsize), np.random.randint(0, img.shape[1]-minsize))  # upper left corner of cutout rectangle
        rwidth, rheight = np.random.randint(minsize, maxsize), np.random.randint(minsize, maxsize)  # width & height of cutout rectangle
        pt2 = ( min(pt1[0] + rwidth, img.shape[0]-1) , min(pt1[1] + rheight, img.shape[1]-1)  )   # keep rectangle bounded in image
        const_value = np.random.uniform( colormin, colormax )
        img[ pt1[0]:pt2[0], pt1[1]:pt2[1], : ] = const_value
    # no return because image is altered in place

'''NEVER USED, ACTUALLY
def cutout_image(img, metadata, file_prefix, num_regions):
    """
     like above, but used for image augmentation
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
'''

def salt_n_pepa_inplace(img, salt_vs_pepper = 0.2, amount=0.004):
    """
    Adds (even more) black & white dots to image
    modified from https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
    Note: operates IN PLACE on one 'image'.  img is really a numpy array
    """
    push_it = np.random.choice(['good','not good'])   # randomly do it or don't do it
    if push_it != 'good':
        return              # abort and leave image unchanged

    salt_color, pepper_color = np.max(img), np.min(img)
    #print("Salt & Pepa's here, and we're in effect!")
    num_salt = np.ceil(amount * img.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))

    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[0:2]]
    img[coords[0], coords[1], :] = salt_color

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[0:2]]
    img[coords[0], coords[1], :] = pepper_color

    # no return because img is altered in place



def rotate_image(img, metadata, file_prefix, rot_angle, rot_origin=None):
    # note that cv2 sometimes reverses what normal humans consider x and y coordinates
    new_img = img.copy()
    if (0 == rot_angle):     # do nothing
        return new_img, list(metadata), file_prefix

    height, width, channels = img.shape
    if (rot_origin is None):                            # if not specified, rotate about image center
        rot_origin = (width/2, height/2)
    rot_matrix = cv2.getRotationMatrix2D(rot_origin, rot_angle, 1.0) # used for image and for changing cx, cy
    new_img = cv2.warpAffine(new_img, rot_matrix, (width,height))

    new_metadata = []
    for md in metadata:
        [cx, cy, a, b, angle, rings] =  md
        angle += rot_angle
        angle = cleanup_angle( angle )
        myPoint = np.transpose( np.array( [ cx, cy, 1] ) )
        newPoint = np.matmul ( rot_matrix, myPoint)
        cx, cy = int(round(newPoint[0])), int(round(newPoint[1]))
        new_metadata.append( [cx, cy,  a, b, angle, rings] )

    new_prefix = file_prefix[:] + "_r{:>.2f}".format(rot_angle)
    return new_img, new_metadata, new_prefix


def invert_image(img, metadata, file_prefix):
    # inverts color; unused; not appropos to SVP dataset
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
