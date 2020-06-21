#! /usr/bin/env python

import numpy as np
import sys
sys.path.append("spnet/")
sys.path.append("../spnet/")
from diagnostics import compute_iou

def test_compute_iou():
    # make up two ellipes
    Y_true = (100,  140, 120,  60,     90, 0, 10.3)  # ellipse a
    Y_pred = (120,  123, 120,  60, 149.97, 0,  7.8)  # ellips b
    #iou = evaluate_spnet.compute_iou(Y_true, Y_pred)
    iou = compute_iou(Y_true, Y_pred)
    np.testing.assert_equal(iou, 0.44227983107795693) # force an assertion for the test
    return iou


if __name__ == '__main__':
    # current setup: testing a couple pre-defined ellipses
    iou = test_compute_iou()
    print("IOU score = ",iou)
