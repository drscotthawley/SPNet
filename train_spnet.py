#! /usr/bin/env python3
from __future__ import print_function

# disable FutureWarnings from numpy re. tensorflow
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer

# other imports
import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint,  EarlyStopping, TensorBoard
import time
import sys

from distutils.version import LooseVersion
import PIL
import random
from spnet import models, utils, multi_gpu, diagnostics, callbacks
import spnet.config as cf
from predict_spnet import predict_network
from evaluate_spnet import evaluate_network

import os


def train_network(weights_file="weights.hdf5", datapath=".", fraction=1.0, batch_size=32, \
        epochs=30, pred_grid=[6,6,2], noaugment=False, log_dir=".", lr_max=4e-5,
        freeze_fac=0.7, frozen_epochs=4, random_seed=1):
    # for deterministic results (TODO: remove later for more general testing)
    np.random.seed(random_seed)
    from tensorflow import set_random_seed
    set_random_seed(random_seed)

    print("pred_grid = ",pred_grid)

    # options for how we run the model
    model_type = cf.model_type

    # Load data
    trainpath = datapath + "/Train/"
    valpath = datapath + "/Val/"
    X_train, Y_train, train_file_list, pred_shape = utils.build_dataset(path=trainpath, \
        load_frac=fraction, set_means_ranges=True, batch_size=batch_size, pred_grid=pred_grid)

    X_val, Y_val, val_file_list, pred_shape  = utils.build_dataset(path=valpath, load_frac=1.0, \
        set_means_ranges=False, batch_size=batch_size, pred_grid=pred_grid)

    print("Seting up NN model.  model_type = ",cf.model_type)
    parallel=False
    model, serial_model = models.setup_model(X_train, Y_train[0].size, no_cp_fatal=False,
        weights_file=weights_file, parallel=parallel, freeze_fac=freeze_fac)

    # Set up callbacks
    myprogress = callbacks.MyProgressCallback(X_val=X_val, Y_val=Y_val, val_file_list=val_file_list, log_dir=log_dir, pred_shape=pred_shape)
    checkpointer = callbacks.ParallelCheckpointCallback(model, filepath=weights_file, save_every=5, dir=log_dir)
    lr_sched = callbacks.OneCycleScheduler(lr_max=lr_max, n_data_points=X_train.shape[0], epochs=epochs, batch_size=batch_size, verbose=1)
    #tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)  #Tensorfboard can be a memory hog.
    #earlystopping = EarlyStopping(patience=5)
    callback_list = [myprogress, checkpointer, lr_sched]# , earlystopping]#, tensorboard]
    if not noaugment:
        print("Adding callback for augment on the fly")
        aug_on_fly = callbacks.AugmentOnTheFly(X_train, Y_train, aug_every=1)
        callback_list.append(aug_on_fly)

    later_epochs = epochs - frozen_epochs

    # early training with partially-frozen pre-trained model
    if (frozen_epochs > 0) and (freeze_fac > 0.0):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=frozen_epochs, shuffle=True,
                  verbose=1, validation_data=(X_val, Y_val), callbacks=callback_list)
    if (freeze_fac > 0.0):
        model = models.unfreeze_model(model, X_train, Y_train, parallel=parallel)

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
    parser = argparse.ArgumentParser(description="trains network on training dataset",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
     # greater batch size runs faster but may yield Out Of Memory errors
     # also note that small batches may yield better generalization: https://arxiv.org/pdf/1609.04836.pdf
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size to use', default=16)
    parser.add_argument('-d', '--datapath', help='Directory with images in Train/ and Val/ subdirs', default="./")
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to run', default=100)
    parser.add_argument('-f', '--fraction', type=float, help='Fraction of dataset to use (for quick testing: -f 0.05)', default=1.0)
    parser.add_argument('--freeze_fac', type=float, help='Fraction of base model (e.g. Xception) to freeze', default=0.0)
    parser.add_argument('--frozen_epochs', type=int, help='Number of starting epochs to run while base model is frozen', default=0)
    parser.add_argument('-g', '--grid', help='Shape of predictor grid', default="6x6x2")
    parser.add_argument('-w', '--weights', help='Weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-l', '--lrmax', type=float, help='Maximum learning rate value', default=4e-5)
    parser.add_argument('-n', '--noaugment', action='store_true', help="don't augment on the fly")
    #Started forgetting why I was doing a run, so added a name param:
    parser.add_argument('--name', help='Descriptive name of the run, prepended to the log directory name', default='')
    parser.add_argument('-r', '--random_seed', type=int, help="Random seed value", default=1)

    args = parser.parse_args()
    print("Command line ~= \n",' '.join(s for s in sys.argv))
    print("args = ",args)

    pred_grid = [int(i) for i in args.grid.split('x')]  # convert string to shape

    now = time.strftime("%c").replace('  ','_').replace(' ','_')   # date, with no double spaces or spaces
    log_dir = './logs/'+args.name+'_'+now if args.name else './logs/'+now
    print("Logging will go to ",log_dir)

    print("\n----------------------------\nStarting training...")
    model = train_network(weights_file=args.weights, datapath=args.datapath, fraction=args.fraction, \
        batch_size=args.batch_size, epochs=args.epochs, pred_grid=pred_grid, noaugment=args.noaugment,
        log_dir=log_dir, lr_max=args.lrmax, freeze_fac=args.freeze_fac, frozen_epochs=args.frozen_epochs,
        random_seed=args.random_seed)

    # run model evaluation
    print("\n----------------------------\nStarting model evaluation...")
    testpath = args.datapath+'/Test/'
    if not os.path.isdir(testpath):
        testpath = args.datapath+'/Val/'
    #X_test, Y_test, test_file_list, pred_shape  = utils.build_dataset(path=testpath, load_frac=1.0, \
    #    set_means_ranges=False, batch_size=args.batch_size, pred_grid=pred_grid)
    evaluate_network(model=model, weights_file="", datapath=testpath, fraction=1.0, log_dir="logs/Evaluation/",
        batch_size=args.batch_size, pred_grid=pred_grid, set_means_ranges=False)

    # make predictions on Zooniverse dataset
    print("\n----------------------------\nStarting Zooniverse predictions...")
    predict_network(weights_file="", fraction=args.fraction,
        log_dir='logs/Predicting/', batch_size=args.batch_size, model=model, X_pred='')

    weights2name = "final_"+args.weights
    print("Just to be sure: Saving model to",weights2name)
    model.save_weights(weights2name)

    print("And saving full model too")
    model.save("full_model.h5")
    print("SPNet execution completed.")
    os.system(f'cp nohup.out *.h5 *.hdf5 {log_dir}') # copy things to the log dir
