#! /usr/bin/env python3

"""
Some diagnostic utilities such as computing the Intersection-Over-Union Score and mAP

Precision measures false positive rate: ratio of true object detections to the total number of objects that the detector predicted. 

"""

import numpy as np
import cv2
from utils import *


def draw_filled_ellipse(img, center, axes, angle):
    return draw_ellipse(img, center, axes, angle, thickness=-1, color=(255))

def create_ellipse_image(args, nx=512, ny=384):
    '''
    args = (cx, cy, a, b, angle, noobj, rings)
    '''
    img = np.zeros( (ny, nx, 1), np.uint8)
    img = draw_filled_ellipse(img, args[0:2], args[2:4], args[4])
    return img


def compute_iou(args_a, args_b, display=True):
    """
    Note: this operates on one pair of ellipses at a time
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

if __name__ == '__main__':

    Y_true = (100,  140,   120,  60,   90,   0, 10.3)
    Y_pred = (120,  123,    120,   60,   149.97,   0,  7.8)
    iou = compute_iou(Y_true, Y_pred)
    print("IOU score = ",iou)
