import numpy as np
import sys
sys.path.append("..")
import evaluate_spnet

def test_compute_iou():
    Y_true = (100,  140, 120,  60,     90, 0, 10.3)
    Y_pred = (120,  123, 120,  60, 149.97, 0,  7.8)
    iou = evaluate_spnet.compute_iou(Y_true, Y_pred)
    np.testing.assert_equal(iou, 0.44227983107795693)
    return iou


if __name__ == '__main__':
    # current setup: testing a couple pre-defined ellipses
    iou = test_compute_iou()
    print("IOU score = ",iou)
