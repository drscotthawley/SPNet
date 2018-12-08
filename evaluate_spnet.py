#! /usr/bin/env python3

# Pulls up some image files and predicts ellipses & ring-counts for them.

import numpy as np
import cv2
import random
import os
import time
import keras
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback
from keras.models import load_model
import time
from distutils.version import LooseVersion
from operator import itemgetter
import PIL
import io
from models import *
from utils import *
from diagnostics import compute_iou


# Main routine
def evaluate_network(weights_file="", datapath="", fraction=1.0, log_dir="", batch_size=32):
    np.random.seed(1)

    print("Getting data..., fraction = ",fraction)
    testpath="Test/"
    X_test, Y_test, test_file_list, pred_shape  = build_dataset(path=testpath, \
        load_frac=fraction, set_means_ranges=True, batch_size=batch_size, shuffle=False)


    print("Loading model from",weights_file)
    model, serial_model = setup_model(X_test, try_checkpoint=True, no_cp_fatal=True, \
        weights_file=weights_file, parallel=False)


    m = X_test.shape[0]
    print("    Predicting... (m = ",m," frames in Test set)",sep="")
    start_time = time.time()
    Y_pred = model.predict(X_test, batch_size=batch_size)
    elapsed = time.time() - start_time
    print("    ...elapsed time to predict = ",elapsed,"s.   FPS = ",m*1.0/elapsed)

    print("    Drawing sample ellipse images...")
    make_sure_path_exists(log_dir)
    Yt, Yp = denorm_Y(Y_test), denorm_Y(Y_pred)
    show_pred_ellipses(Yt, Yp, test_file_list, log_dir=log_dir, out_csv=log_dir+'hawley_spnet.csv')

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="tests network on test dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-d', '--datapath', #type=argparse.string,
        help='Test dataset directory with list of classes', default="Test/")
    parser.add_argument('-f', '--fraction', type=float,
        help='Fraction of dataset to use', default=1.0)
    parser.add_argument('-l', '--logdir',
            help='Directory of log files', default='logs/Testing/')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size to use', default=32)

    args = parser.parse_args()
    model = evaluate_network(weights_file=args.weights, datapath=args.datapath, fraction=args.fraction, log_dir=args.logdir, batch_size=args.batch_size)
