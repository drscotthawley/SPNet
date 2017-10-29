#! /usr/bin/env python3
from __future__ import print_function

import numpy as np

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.applications import Xception, VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.utils.generic_utils import CustomObjectScope
from keras.applications.inception_v3 import preprocess_input
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
from keras.optimizers import SGD, Adam
from os.path import isfile
from utils import *
from multi_gpu import *

def instantiate_model(X, Y, freeze_fac=1.0):
    # Pick a pre-trained model, but leave the "top" off.

    input_tensor = Input(shape=X[0].shape)
    weights = None#'imagenet'    # None or 'imagenet'    If you want to use imagenet, X[0].shape[2] must = 3
    #base_model = VGG16(weights=weights, include_top=False, input_tensor=input_tensor)              # same for all inputs
    #--nope base_model = Xception(weights=weights, include_top=False, input_tensor=input_tensor)   # yield same #s for all inputs
    #base_model = InceptionV3(weights=weights, include_top=False, input_tensor=input_tensor)       # Works well, fast
    #base_model = InceptionResNetV2(weights=weights, include_top=False, input_tensor=input_tensor)  # works ok, big, slow.  doesn't play well with "unfreeze" in multi-gpu setting
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        base_model = MobileNet(input_shape=X[0].shape, weights=weights, include_top=False, input_tensor=input_tensor)       # Works well, VERY FAST! Needs img size 224x224

    top_model = Sequential()        # top_model gets tacked on to pretrained model
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(Y[0].size,name='FinalOutput'))      # Final output layer

    #top_model.load_weights('bootlneck_fc_model.h5')
    model = Model(input= base_model.input, output= top_model(base_model.output))

    # set the first N layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # for fine-tuning, "freeze" most of the pre-trained model, and then unfreeze it later
    num_layers = len(base_model.layers)
    freeze_layers = int(num_layers * freeze_fac)
    if (freeze_layers > 0 ):
        print("Freezing ",freeze_layers,"/",num_layers," layers of base_model")
        for i in range(freeze_layers):
            base_model.layers[i].trainable = False

    return model

def yolo_loss(Ypred,Ytrue):
    # YOLO essentially uses MSE on everything.
    #  - The authors remark (in rebuttal to NIPS reviewers) that "We tried usign softmax for classes early on but found it was not [as] effective as Euclidean loss"
    #  - They actually predict on the square root of their bounding box sizes to make small boxes fit tighter than large boxes
    #  - The apply different regularization parameters to different quantities (i.e. they weight them differently)
    # Modification: We're going to use MSE on the angle of the ellipse BUT we'll regularize it by the difference between ellipse's semimajor & minor axis sizes
    #     because we don't care what the orientation angle is for a circle.  We want long-skinny ellipses better aligned than 'fat' ellipses
    diff = (Y_pred - Y_true)
    se = diff**2

    # regularization
    lambda_centroid = 1.0
    lambda_size = 1.0
    lambda_angle = 1.0
    lambda_noobj = 1.0

    se[ind_cx:ind_cy] *= lambda_centroid
    se[ind_semi_a:ind_semi_b] *= lambda_size
    se[ind_angle] *= lambda_angle*((Y_true[ind_semi_a] - Y_true[ind_semi_b])/Y_true[ind_semi_a])**2  # regularize by difference, normalize by size, square to keep positive
    se[ind_noobj] *= lambda_noobj

    loss = np.mean(se)

    # gradients
    gradients = np.mean(2*diff)    # TODO: add lambda's to gradient

    return loss, gradients


def setup_model(X, Y, nb_layers=4, try_checkpoint=True,
    no_cp_fatal=False, weights_file='weights.hdf5', freeze_fac=0.75, opt='adam', parallel=True):

    model = None
    from_scratch = True
    # Initialize weights using checkpoint if it exists.
    if (try_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(weights_file) ):
            print ('Weights file detected. Loading from ',weights_file)
            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                if parallel:   # if we're loading from something that was already a multi-gpu model
                    multi_model = load_model(weights_file, custom_objects={"tf": tf})
                    model = multi_model.layers[-2]   # strip parallel part, to be added back in later
                else:
                    model = load_model(weights_file, custom_objects={"tf": tf})

                #model = make_parallel(model, 2)

            from_scratch = False
        else:
            if (no_cp_fatal):
                raise Exception("*** No weights file detected; can't do anything.  Aborting.")
            else:
                print('    No weights file detected, so starting from scratch.')

    if from_scratch:
        #model = MLP(X, num_outputs=Y.shape[1], nb_layers=nb_layers)
        #model = MyCNN_Keras2(X, num_outputs=Y.shape[1], nb_layers=nb_layers)
        model = instantiate_model(X, Y, freeze_fac=freeze_fac) # start by freezing

    if parallel:
        model = make_parallel(model, 2)    # easier to "unfreeze" later if we leave it in serial

    loss = custom_loss # custom_loss or 'mse'
    model.compile(loss=loss, optimizer=opt)

    model.summary()

    return model


def unfreeze_model(model, X, Y, freeze_fac=0.0, parallel=True):
    print("Unfreezing Model: make a new identical model, then copy the layer weights.")
    new_model = instantiate_model(X, Y, freeze_fac=freeze_fac)  # identical spec as original model, just not setting 'trainable=False'

    if parallel:
        single_model = model.layers[-2]         # strip off parallel part
        new_model.set_weights( single_model.get_weights() )     # copy layer weights
    else:
        new_model.set_weights( model.get_weights() )     # copy layer weights

    if parallel:
        new_model = make_parallel(new_model, 2)          # kick it in to high gear

    opt = Adam()#lr=0.0001)

    new_model.compile(loss='mse', optimizer=opt)
    print("  ...finished un-freezing model")
    return new_model
