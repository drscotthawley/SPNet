#! /usr/bin/env python3

# Pulls up some image files and predicts ellipses & ring-counts for them.

# Does not perform any kind of scoring or evaluation.  The assumption is
# that annotations for these images my not exist. 

# disable FutureWarnings from numpy re. tensorflow
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer

# other imports
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
import spnet.config as cf


default_image_dir = '/home/shawley/datasets/zooniverse_steelpan/'

def predict_network(weights_file="spnet.model", datapath=default_image_dir, fraction=1.0,
    log_dir='logs/Predicting/', batch_size=16, model=None, X_pred=''):

    if '' == X_pred:   # None doesn't cut it
        print(f"Getting data from {datapath}, fraction = {fraction}.")
        # options for how we run the model
        model_type = cf.model_type
        if model_type == 'simple':
            grayscale = False
            force_dim = 224
        elif cf.model_type == 'big':
            grayscale = True  # throw out any RGB components in image file, just keep R
            force_dim = None  # Don't resize input images. May need to decrease batch size to fit in VRAM
        else:
            grayscale = True
            force_dim = 331

        img_file_list = sorted(glob.glob(datapath+'/*.png'))
        if len(img_file_list) == 0:
            img_file_list += sorted(glob.glob(datapath+'/*.bmp'))

        total_files = len(img_file_list)
        total_load = int(total_files * fraction)
        if (batch_size is not None):                # keras gets particular: dataset size must be mult. of batch_size
            total_load = nearest_multiple( total_load, batch_size)
        print("      Total files = ",total_files,", going to load total_load = ",total_load)
        X_pred, img_dims = build_X(total_load, img_file_list, force_dim=force_dim, grayscale=grayscale)

        print("")

    if model==None:
        print("Loading model from",weights_file)
        if ('.hdf5' in weights_file):
            print("   Defining model, then loading weights")
            model, serial_model = setup_model(X_pred, try_checkpoint=True, no_cp_fatal=True,
                weights_file=weights_file, parallel=False, freeze_fac=0.0, quick_setup=True)
        else:
            print("   Loading whole model")
            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'custom_loss':custom_loss, 'tf':tf}):   # older keras
                model = keras.models.load_model(weights_file)


    m = X_pred.shape[0]
    print("    Predicting... (m = ",m," frames in dataset)",sep="")
    start_time = time.time()
    Y_pred = model.predict(X_pred, batch_size=batch_size)
    elapsed = time.time() - start_time
    print("    ...elapsed time to predict = ",elapsed,"s.   FPS = ",m*1.0/elapsed)

    print("    Drawing ellipse images...")
    make_sure_path_exists(log_dir)
    pred_shape = [6,6,2,cf.vars_per_pred]  # TODO: un-hard-code this
    setup_means_and_ranges(pred_shape)
    Yp = denorm_Y(Y_pred)   # TODO I think the error is here
    show_pred_ellipses(Yp, Yp, img_file_list, num_draw=m, log_dir=log_dir, \
        out_csv=log_dir+'hawley_spnet.csv', show_true=False)

    return model


if __name__ == '__main__':
    np.random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description="tests network on test dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="spnet.model")
    parser.add_argument('-d', '--datapath', #type=argparse.string,
        help='Dataset directory with list of images', default=default_image_dir)
    parser.add_argument('-f', '--fraction', type=float,
        help='Fraction of dataset to use', default=1.0)
    parser.add_argument('-l', '--logdir',
            help='Directory of log/output files', default='logs/Predicting/')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size to use', default=16)

    args = parser.parse_args()
    model = predict_network(weights_file=args.weights, datapath=args.datapath, fraction=args.fraction, log_dir=args.logdir, batch_size=args.batch_size)
