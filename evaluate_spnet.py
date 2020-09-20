#! /usr/bin/env python3

# Additional scoring routines for trained models
# TODO: Note this whole thing is almost the same code as predict_spnet.py

# disable FutureWarnings from numpy re. tensorflow
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer

# Main imports
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
from spnet.diagnostics import compute_iou
import spnet.config as cf


# Main routine
def evaluate_network(weights_file="", datapath="Test/", fraction=1.0, log_dir="", batch_size=32):
    np.random.seed(1)

    print("Getting data..., fraction = ",fraction)
    testpath = datapath
    # options for how we run the model
    model_type = cf.model_type
    X_test, Y_test, test_file_list, pred_shape  = build_dataset(path=testpath, \
        load_frac=fraction, set_means_ranges=True, batch_size=batch_size, shuffle=False)


    print("Loading model from",weights_file)
    model, serial_model = setup_model(X_test, try_checkpoint=True, no_cp_fatal=True, \
        weights_file=weights_file, parallel=False, quick_setup=True)


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

    # Comput metrics
    ring_miscounts, total_obj, pix_err, ipem = diagnostics.calc_errors(Yp, Yt)
    class_acc = (total_obj-ring_miscounts)*1.0/total_obj*100
    print('Mean pixel error =',np.mean(pix_err))
    print("Num ring miscounts = ",ring_miscounts,' / ',total_obj,'.   = ',class_acc,' % class. accuracy',sep="")

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
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size to use', default=16)

    args = parser.parse_args()
    model = evaluate_network(weights_file=args.weights, datapath=args.datapath, fraction=args.fraction, log_dir=args.logdir, batch_size=args.batch_size)

    weights2name = "eval_end_weights.hdf5"
    print("Saving model to",weights2name)
    model.save_weights(weights2name)
