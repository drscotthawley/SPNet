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
from spnet.models import *
from spnet.utils import *



def predict_network(weights_file="", datapath="", fraction=1.0, log_dir="", batch_size=32):
    np.random.seed(1)

    print("Getting data..., fraction = ",fraction)
    img_file_list = sorted(glob.glob(datapath+'/*.png'))
    total_files = len(img_file_list)
    total_load = int(total_files * fraction)
    if (batch_size is not None):                # keras gets particular: dataset size must be mult. of batch_size
        total_load = nearest_multiple( total_load, batch_size)
    print("      Total files = ",total_files,", going to load total_load = ",total_load)
    X_pred, img_dims = build_X(total_load, img_file_list, force_dim=224, grayscale=False)
    print("")


    print("Loading model from",weights_file)
    model, serial_model = setup_model(X_pred, try_checkpoint=True, no_cp_fatal=True, \
        weights_file=weights_file, parallel=False)


    m = X_pred.shape[0]
    print("    Predicting... (m = ",m," frames in dataset)",sep="")
    start_time = time.time()
    Y_pred = model.predict(X_pred, batch_size=batch_size)
    elapsed = time.time() - start_time
    print("    ...elapsed time to predict = ",elapsed,"s.   FPS = ",m*1.0/elapsed)

    print("    Drawing ellipse images...")
    make_sure_path_exists(log_dir)
    pred_shape = [6,6,2,8]  # TODO: un-hard-code this
    setup_means_and_ranges(pred_shape)
    Yp = denorm_Y(Y_pred)
    show_pred_ellipses(Yp, Yp, img_file_list, num_draw=m, log_dir=log_dir, \
        out_csv=log_dir+'hawley_spnet.csv', show_true=False)

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="tests network on test dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-d', '--datapath', #type=argparse.string,
        help='Dataset directory with list of images', default="../datasets/zooniverse_steelpan/")
    parser.add_argument('-f', '--fraction', type=float,
        help='Fraction of dataset to use', default=1.0)
    parser.add_argument('-l', '--logdir',
            help='Directory of log/output files', default='logs/Predicting/')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size to use', default=32)

    args = parser.parse_args()
    model = predict_network(weights_file=args.weights, datapath=args.datapath, fraction=args.fraction, log_dir=args.logdir, batch_size=args.batch_size)
