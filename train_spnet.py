#! /usr/bin/env python3
from __future__ import print_function
import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint,  EarlyStopping, TensorBoard
import time

from distutils.version import LooseVersion
import PIL
from spnet import models, utils, multi_gpu, diagnostics, callbacks
import random


def train_network(weights_file="weights.hdf5", datapath="Train/", fraction=1.0, batch_size=32, \
        epochs=30, pred_grid=[6,6,2], noaugment=False):
    # for deterministic results (TODO: remove later for more general testing)
    np.random.seed(1)
    from tensorflow import set_random_seed
    set_random_seed(1)

    print("pred_grid = ",pred_grid)

    # options for how we run the model
    model_type = 'bigger'#'monolithic' or 'complex'
    if model_type == 'simple':
        grayscale = False
        force_dim = 224
    else:
        grayscale = True
        force_dim = 331

    # Load data
    X_train, Y_train, train_file_list, pred_shape = utils.build_dataset(path=datapath, \
        load_frac=fraction, set_means_ranges=True, batch_size=batch_size, pred_grid=pred_grid, \
        grayscale=grayscale, force_dim=force_dim)
    valpath="Val/"
    X_val, Y_val, val_file_list, pred_shape  = utils.build_dataset(path=valpath, load_frac=1.0, \
        set_means_ranges=False, batch_size=batch_size, pred_grid=pred_grid,\
        grayscale=grayscale, force_dim=force_dim)

    print("Seting up NN model...")
    parallel=True
    freeze_fac=0.0
    model, serial_model = models.setup_model(X_train, Y_train[0].size, no_cp_fatal=False, weights_file=weights_file, parallel=parallel, \
        freeze_fac=freeze_fac, model_type=model_type)

    # Set up callbacks
    now = time.strftime("%c").replace('  ','_').replace(' ','_')   # date, with no double spaces or spaces
    log_dir='./logs/'+now
    myprogress = callbacks.MyProgressCallback(X_val=X_val, Y_val=Y_val, val_file_list=val_file_list, log_dir=log_dir, pred_shape=pred_shape)
    checkpointer = callbacks.ParallelCheckpointCallback(serial_model, filepath=weights_file, save_every=10)
    lr_sched = callbacks.OneCycleScheduler(lr_max=5e-4, n_data_points=X_train.shape[0], epochs=epochs, batch_size=batch_size, verbose=1)
    #Tenosrboard can be a memory hog. tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    #earlystopping = EarlyStopping(patience=5)
    callback_list = [myprogress, checkpointer, lr_sched]# , earlystopping]#, tensorboard]
    if not noaugment:
        aug_on_fly = callbacks.AugmentOnTheFly(X_train, Y_train, aug_every=2)
        callback_list.append(aug_on_fly)

    frozen_epochs = 0;  # how many epochs to first run with last layers of model frozen
    later_epochs = epochs - frozen_epochs

    # early training with partially-frozen pre-trained model
    if (frozen_epochs > 0) and (freeze_fac > 0.0):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=frozen_epochs, shuffle=True,
                  verbose=1, validation_data=(X_val, Y_val), callbacks=callback_list)
    if (freeze_fac > 0.0):
        model = unfreeze_model(model, X_train, Y_train, parallel=parallel)

    # main training block
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=later_epochs, shuffle=True,
              verbose=1, validation_data=(X_val, Y_val), callbacks=callback_list)

    #serial_model.save_weights(weights_file)
    return model



if __name__ == '__main__':
    seed = 1   # set to None to remove reproducibility
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    import argparse
    parser = argparse.ArgumentParser(description="trains network on training dataset")
     # greater batch size runs faster but may yield Out Of Memory errors
     # also note that small batches may yield better generalization: https://arxiv.org/pdf/1609.04836.pdf
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size to use', default=36)
    parser.add_argument('-d', '--datapath', help='Train dataset directory with list of classes', default="Train/")
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to run', default=30)
    parser.add_argument('-f', '--fraction', type=float, help='Fraction of dataset to use (for quick testing: -f 0.05)', default=1.0)
    parser.add_argument('-g', '--grid', help='Shape of predictor grid', default="6x6x2")
    parser.add_argument('-w', '--weights', help='Weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-n', '--noaugment', action='store_true', help="don't augment on the fly")

    args = parser.parse_args()
    print("args = ",args)

    pred_grid = [int(i) for i in args.grid.split('x')]  # convert string to shape

    model = train_network(weights_file=args.weights, datapath=args.datapath, fraction=args.fraction, \
        batch_size=args.batch_size, epochs=args.epochs, pred_grid=pred_grid, noaugment=args.noaugment)

    # TODO: Score the model against Test dataset
    print("SPNet execution completed.")
