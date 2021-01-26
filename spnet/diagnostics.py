#! /usr/bin/env python3
"""
Some diagnostic utilities such as computing the Intersection-Over-Union Score and mAP
"""
import numpy as np
import cv2
import spnet.config as cf
from spnet import utils




def calc_errors(Yp, Yt):
    """
    Calc errors in ring counts
      Yp = model prediction
      Yt = true value
      Yp & Yt have already been 'denormalized' at this point

      Not: index = 2 is where the ring count is stored.
    """
    max_pred_antinodes = int(Yt.shape[1]/cf.vars_per_pred)
    diff = Yp - Yt
    # pixel error in antinode centers
    pix_err = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
    ipem = np.argmax(pix_err)               # index of pixel error maximum

    # number of ring miscounts
    ring_miscounts, ring_truecounts = 0, 0
    total_obj = 0         # total is the number of TRUE objects
    false_obj_pos = 0     # false positives (predicting obj where shouldn't be)
    false_obj_neg = 0     # false negatives (should be object but none predicted)
    true_obj_pos, true_obj_neg = 0, 0   # sanity check: how often do we get these right?
    for j in range(Yt.shape[0]):
        for an in range(max_pred_antinodes):
            ind = cf.ind_rings + an * cf.vars_per_pred
            rings_t = int(round(Yt[j,ind]))
            i_noobj = cf.ind_noobj + an * cf.vars_per_pred
            if (0 == int(round(Yt[j,i_noobj])) ):   # Is there supposed to be an object there? If so, count the rings
                total_obj += 1
                if (0 == int(round(Yp[j,i_noobj]))): # yay, we correctly predicted an object here
                    true_obj_pos += 1
                    # But did we count the rings correctly?
                    # note: when Yt ring counts are ints, the following two if statements are equivalnt. When Yt is not an int, they yield different results
                    if (np.abs(Yt[j,ind] - Yp[j,ind]) > 0.5): # predicted one but ring count is off. should be within 0.5 (as if rounding to int) so 1.4 vs 1.6 is ok
                    #if (int(round(Yp[j,ind])) != rings_t): # Example of what we don't want: if ring counts do not round to the same int, so 1.4 vs. 1.6 gets regarded as an error
                        ring_miscounts += 1
                    else:
                        ring_truecounts += 1
                else:                                # supposed to be object but we didn't predict one
                    false_obj_neg += 1

            else:                                     # not supposed to be an object there
                if (0 == int(round(Yp[j,i_noobj]))):  # ...but oops, we predicted one
                    false_obj_pos += 1
                else:
                    true_obj_neg += 1                 # yay, we correctly...didn't predict anything.

    return ring_miscounts, ring_truecounts, total_obj, false_obj_pos, false_obj_neg, true_obj_pos, true_obj_neg, pix_err, ipem



#--------- Routines for computing Intersection over Union (IoU)
def draw_filled_ellipse(img, center, axes, angle):
    return utils.draw_ellipse(img, center, axes, angle, thickness=-1, color=(255))

def create_ellipse_image(args, nx=512, ny=384):
    '''
      If it's not supposed to exist, then this returns an image of all zeros
      nx, ny = image dimensions, resolution
    '''
    img = np.zeros( (ny, nx, 1), np.uint8)   # note we reverse nx & ny b/c numpy
    (cx, cy, a, b, cos2t, sin2t, noobj, rings) = args

    if (noobj < 0.5): # check that ellipse is supposed to exist before drawing anything
        angle = np.rad2deg( np.arctan2(sin2t,cos2t)/2.0 )
        img = draw_filled_ellipse(img, [cx,cy], [a,b], angle)

    return img


def compute_iou(args_p, args_t, display=False):
    """
    Note: this operates on one pair of ellipses at a time
       args_a:  Parameters for the predicted ellipse, where
          args = (cx, cy, a, b, cos2t, sin2t, noobj, rings)
       args_b:  Parameters for the true ellipse
    If only one of the ellipses does not exist, this will return zero (intersection=0, union!=0)

    # Since the true values come with "default" values for thing, they are denoted by
        noobj = 1.
    If both ellipses don't exist (nothing supposed to be there, nothing predicted there), then
      then this returns a code of -1
    """
    # args = (cx, cy, a, b, angle, noobj, rings)
    if args_t[-2] > 0.99: # todo just for now
        return -1
    #print("    compute_iou: args_p =",args_p)
    #print("    compute_iou: args_t =",args_t)
    img_p = create_ellipse_image(args_p)
    img_t = create_ellipse_image(args_t)
    img_i = cv2.bitwise_and(img_p, img_t)   # intersection image
    img_u = cv2.bitwise_or(img_p, img_t)    # union image
    num_i, num_u = cv2.countNonZero(img_i) , cv2.countNonZero(img_u)
    #print("          num_i, num_u =",num_i, num_u)
    iou = 0
    if (num_i == 0) and (num_u == 0):
        iou = -1
    elif num_u > 0:
        iou = num_i / num_u
    else:                  # this is only reached if there are no ellipses defined at all
        assert False,"Error in compute_iou: "

    if display:
        cv2.imshow('img_a', img_p)
        cv2.imshow('img_b', img_t),
        cv2.imshow('intersection', img_i)
        cv2.imshow('union', img_u), cv2.waitKey(0)
        cv2.destroyAllWindows()
    return iou
#--------- End of routines for IOU


# Precision measures false positive rate: ratio of true object detections to the total number of objects that the detector predicted.
def precision(Yp, Yt, thresh=0.5):
    tp_count = 0 # true positive: predicted it was supposed to, with iou > thresh.
    fp_count = 0 # false positive: shouldn't have predicted anythihg but did
    fn_count = 0 # false neg: should have predicted something but didn't
    for i in range(Yp.shape[0]): # loop over images
        for j in np.arange(0, Yp.shape[1], cf.vars_per_pred): # loop over grid-predictors
            args_p = Yp[i,j:j+cf.vars_per_pred]       # predicted values
            args_t = Yt[i,j:j+cf.vars_per_pred]       # true / ground-truth values
            iou = compute_iou(args_p, args_t)
            if iou < 0:
                continue  # skip what's below but keep going in for loops
            #print("i, j, iou = ",i,j,iou)
            if (iou > thresh): # this is a "hit"!
                tp_count += 1
            elif (args_p[-2] < 0.5) and (args_t[-2] >= 0.5): # the -2 element is "noobj", i.e. (opposite of) existence
                fp_count += 1                         # increment false negs:
            elif (args_p[-2] >= 0.5) and (args_t[-2] < 0.5):
                fn_count += 1

    print("precision: thresh = ",thresh,",tp_count, fp_count, fn_count = ",tp_count, fp_count, fn_count)
    denom = (tp_count + fp_count + fn_count) # total number of ellipses
    #assert denom == Yp.shape[0] # sanity check
    prec =  tp_count / denom
    print("precision: thresh, prec = ",thresh,prec)
    return prec, tp_count, fp_count, fn_count
#--------- End of computing precision

def calc_map(Yp, Yt):
    # calculate Mean Average Precision
    print("\ncalc_map: Calculating mean average precision. Yp.shape[0] =",Yp.shape[0])
    threshes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    prec_tot = 0
    for thresh in threshes:
        prec, tp_count, fp_count, fn_count = precision(Yp, Yt, thresh=thresh)
        prec_tot += prec
    map = prec_tot / len(threshes)
    return map



if __name__ == '__main__':
    from tests import test_diagnostics
    # current setup: testing a couple pre-defined ellipses
    iou = test_diagnostics.test_compute_iou()
    print("IOU score = ",iou)
