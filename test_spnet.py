#! /usr/bin/env python3

# Pulls up some image files and predicts ellipses & ring-counts for them.

import numpy as np
import cv2
import random
import os
import time

import keras
import tensorflow as tf
from tensorflow.contrib.keras.python.keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
import time
from distutils.version import LooseVersion
from operator import itemgetter
import PIL
import io
from models import *
from utils import *

# Define some colors: openCV uses BGR instead of RGB
blue = (255,0,0)
red = (0,0,255)
green = (0,255,0)
white = (255)
black = (0)
grey = (128)





def test_network(weights_file="weights.hdf5", datapath="Train/", fraction=1.0, log_dir="logs/"):
    np.random.seed(1)

    # Get the data
    print("Getting data..., fraction = ",fraction)
    testpath="Test/"
    X_test, Y_test, img_dims, test_file_list  = build_dataset(path=testpath, load_frac=fraction)

    # Instantiate the model
    print("Instantiating model...")
    model = setup_model(X_test, Y_test, no_cp_fatal=True, weights_file=weights_file)

    m = Y_test.shape[0]

    print("    Predicting... (m = ",m," frames in Test set)",sep="")
    start_time = time.time()
    Y_pred = model.predict(X_test)
    elapsed = time.time() - start_time
    print("    ...elapsed time to predict = ",elapsed,"s.   FPS = ",m*1.0/elapsed)

    print("    Drawing sample ellipse images...")
    show_pred_ellipses(Y_test, Y_pred, test_file_list, log_dir=log_dir)

    return model




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="tests network on test dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-c', '--datapath', #type=argparse.string,
        help='Test dataset directory with list of classes', default="Test/")
    parser.add_argument('-f', '--fraction', type=float,
        help='Fraction of dataset to use', default=1.0)
    parser.add_argument('-l', '--logdir',
            help='Directory of log files', default='logs/')

    args = parser.parse_args()
    model = test_network(weights_file=args.weights, datapath=args.datapath, fraction=args.fraction, log_dir=args.logdir)
