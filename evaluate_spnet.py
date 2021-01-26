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
from keras.utils.generic_utils import get_custom_objects, CustomObjectScope
import time
from distutils.version import LooseVersion
from operator import itemgetter
import PIL
import io
from spnet import models, multi_gpu, diagnostics, callbacks
from spnet.utils import *
import spnet.config as cf
from predict_spnet import predict_network


# Main routine
def evaluate_network(model=None, weights_file="", datapath="Test/", fraction=1.0, log_dir="", batch_size=32,
    pred_grid=[6,6,2], set_means_ranges=True):
    np.random.seed(1)

    print("Getting data..., fraction = ",fraction)
    testpath = datapath
    # options for how we run the model
    model_type = cf.model_type

    # the Train data load here is ONLY to normalize means & ranges identically to as they were when training happened
    #X_train, Y_train, train_file_list, pred_shape = build_dataset(path=datapath+'/../Train/', \
    #    load_frac=fraction, set_means_ranges=True, batch_size=batch_size, pred_grid=pred_grid)

    X_test, Y_test, test_file_list, pred_shape  = build_dataset(path=testpath, \
        load_frac=fraction, set_means_ranges=set_means_ranges, batch_size=batch_size, shuffle=False, pred_grid=pred_grid)

    if model is None:
        #print("Setting up model and then loading weights from",weights_file)
        #model, serial_model = models.setup_model(X_test, Y_test[0].size, try_checkpoint=True, no_cp_fatal=True, \
        #    weights_file=weights_file, parallel=False, quick_setup=True, freeze_fac=0)

        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'custom_loss':models.custom_loss, 'tf':tf}):   # older keras
            print("Loading full model from full_model.h5")
            model = load_model("full_model.h5")

    m = X_test.shape[0]      # how many frames (i.e. files) we'll be looking at
    print("    Predicting... (m = ",m," frames in this (Test?) dataset)",sep="")
    start_time = time.time()
    Y_pred = model.predict(X_test, batch_size=batch_size)
    elapsed = time.time() - start_time
    print("    ...elapsed time to predict = ",elapsed,"s.   FPS = ",m*1.0/elapsed)

    if cf.loss_type != 'same':  # convert from logits if needed
        Y_pred[:, cf.ind_noobj::cf.vars_per_pred] = 1.0/(1.0 + np.exp(-Y_pred[:, cf.ind_noobj::cf.vars_per_pred])) # sigmoid

    Yt, Yp = denorm_Y(Y_test), denorm_Y(Y_pred)  # convert from normalized to 'world' values

    print("mAP = ",diagnostics.calc_map(Yp, Yt))

    # Compute metrics
    ring_miscounts, ring_truecounts, total_obj, false_obj_pos, false_obj_neg, true_obj_pos, true_obj_neg, pix_err, ipem = diagnostics.calc_errors(Yp, Yt)
    mistakes = ring_miscounts + false_obj_pos + false_obj_neg
    class_acc = (total_obj-mistakes)*1.0/total_obj*100  # accuracy is based on lack of any mistakes
    print('Mean pixel error =',np.mean(pix_err))
    print("    Ring correct counts = ",ring_truecounts,' / ',total_obj,'.   = ',100*ring_miscounts/total_obj,' % ring-class accuracy',sep="")
    print("         Ring miscounts = ",ring_miscounts,' / ',total_obj,'.   = ',100*ring_miscounts/total_obj,' % ring-miscount rate',sep="")
    print("        False positives = ",false_obj_pos,' / ',total_obj,'.   = ',100*false_obj_pos/total_obj,' % FP rate',sep="")
    print("        False negatives = ",false_obj_neg,' / ',total_obj,'.   = ',100*false_obj_neg/total_obj,' % FN rate',sep="")
    print("         True positives = ",true_obj_pos,' / ',total_obj,'.   = ',100*true_obj_pos/total_obj,' % TP rate',sep="")
    print("         True negatives = ",true_obj_neg,sep="")
    print("    Total Mistakes = ",mistakes,' / ',total_obj,'.   => ',class_acc,' % class. accuracy rate (lack of mistakes)',sep="")

    make_sure_path_exists(log_dir)
    print("    Drawing sample ellipse images...")
    show_pred_ellipses(Yt, Yp, test_file_list, num_draw=m, log_dir=log_dir, out_csv=log_dir+'hawley_spnet.csv')

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="tests network on test dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-d', '--datapath', #type=argparse.string,
        help='Test dataset directory', default="Test/")
    parser.add_argument('-f', '--fraction', type=float,
        help='Fraction of dataset to use', default=1.0)
    parser.add_argument('-l', '--logdir',
            help='Directory to write log files into', default='logs/Testing/')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size to use', default=16)

    args = parser.parse_args()
    model = evaluate_network(weights_file=args.weights, datapath=args.datapath+'/', fraction=args.fraction, log_dir=args.logdir, batch_size=args.batch_size)


    # make predictions on unlabeled dataset
    #predict_network(weights_file="", fraction=args.fraction,
    #    log_dir='logs/Predicting/', batch_size=args.batch_size, model=model, X_pred='')

    weights2name = "eval_end_weights.hdf5"
    print("Saving model to",weights2name)
    model.save_weights(weights2name)
