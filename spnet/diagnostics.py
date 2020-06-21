#! /usr/bin/env python3
"""
Some diagnostic utilities such as computing the Intersection-Over-Union Score (and mAP?)

mAP is typical for classification-based object detectors.
"""
import numpy as np
import cv2
import sys
sys.path.append('/home/shawley/SPNet')
sys.path.append('..')
from spnet import utils
import spnet.config as cf




def calc_errors(Yp, Yv):
    """
    Calc errors in ring counts
      Yp = model prediction
      Yv = true value for validation set
      Yp & Yv have already been 'denormalized' at this point

      Not: index = 2 is where the ring count is stored.
    """
    max_pred_antinodes = int(Yv.shape[1]/cf.vars_per_pred)
    diff = Yp - Yv
    # pixel error in antinode centers
    pix_err = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
    ipem = np.argmax(pix_err)               # index of pixel error maximum

    # number of ring miscounts
    miscounts = 0
    total_obj = 0                           # total is the number of true objects
    for j in range(Yv.shape[0]):
        for an in range(max_pred_antinodes):
            ind = cf.ind_rings + an * cf.vars_per_pred
            rings_t = int(round(Yv[j,ind]))
            i_noobj = cf.ind_noobj + an * cf.vars_per_pred
            if (0 == int(round(Yv[j,i_noobj])) ):   # Is there supposed to be an object there? If so, count the rings
                total_obj += 1
                if (int(round(Yp[j,ind])) != rings_t): # compare integer ring counts
                    miscounts += 1
            elif (int(round(Yv[j,i_noobj])) != int(round(Yp[j,i_noobj]))):  # consider false background as a mistake
                miscounts += 1

    return miscounts, total_obj, pix_err, ipem



#--------- Routines for computing Intersection over Union (IoU)
def draw_filled_ellipse(img, center, axes, angle):
    return utils.draw_ellipse(img, center, axes, angle, thickness=-1, color=(255))

def create_ellipse_image(args, nx=512, ny=384):
    '''
    Inputs: args is a tuple where
        args = (cx, cy, a, b, angle, noobj, rings)
    '''
    img = np.zeros( (ny, nx, 1), np.uint8)
    img = draw_filled_ellipse(img, args[0:2], args[2:4], args[4])
    return img


def compute_iou(args_a, args_b, display=False):
    """
    Inputs:  args_a and args_b are the set of arguments to create_ellipse_image(), see previous routine

    Notes: this operates on one pair of ellipses (a & b) at a time
    If one of the ellipses does not exist, this will return zero (intersection=0, union!=0)
    Any parts of either ellipse that extend off the edge of the image are not included.
    """
    img_a = create_ellipse_image(args_a)
    img_b = create_ellipse_image(args_b)
    img_i = cv2.bitwise_and(img_a,img_b)   # intersection image
    img_u = cv2.bitwise_or(img_a,img_b)    # union image
    num_i, num_u = cv2.countNonZero(img_i) , cv2.countNonZero(img_u)
    if num_u > 0:
        iou = num_i / num_u
    else:                  # this is only reached if there are no ellipses defined at all
        assert False,"Error in compute_iou: "

    if display:
        cv2.imshow('img_a', img_a)
        cv2.imshow('img_b', img_b),
        cv2.imshow('intersection', img_i)
        cv2.imshow('union', img_u), cv2.waitKey(0)
        cv2.destroyAllWindows()
    return iou
#--------- End of routines for IOU


#--------- Compute Precision
# Precision measures false positive rate: ratio of true object detections to the total number of objects that the detector predicted.
#--------- End of computing precision

#--------- Compute Recall
#--------- End of computing recall


if __name__ == '__main__':
    # Here's a simple run of the test
    
    sys.path.append('../tests/')
    import test_diagnostics
    # current setup: testing a couple pre-defined ellipses
    iou = test_diagnostics.test_compute_iou()
    print("IOU score = ",iou)
