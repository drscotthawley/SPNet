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
    model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

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


def setup_model(X, Y, nb_layers=4, try_checkpoint=True,
    no_cp_fatal=False, weights_file='weights.hdf5', freeze_fac=0.75, opt='adam', parallel=True):

    model = instantiate_model(X, Y, freeze_fac=freeze_fac)

    # Initialize weights using checkpoint if it exists.
    if (try_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(weights_file) ):
            print ('Weights file detected. Loading from ',weights_file)
            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'custom_loss':custom_loss, 'tf':tf}):
                if parallel:   # if we're loading from something that was already a multi-gpu model
                    multi_model = load_model(weights_file)
                    loaded_model = multi_model.layers[-2]   # strip parallel part, to be added back in later
                else:
                    loaded_model = load_model(weights_file)
            model.set_weights( loaded_model.get_weights() )
        else:
            if (no_cp_fatal):
                raise Exception("*** No weights file detected; can't do anything.  Aborting.")
            else:
                print('    No weights file detected, so starting from scratch.')

    if parallel:
        model = make_parallel(model)    # easier to "unfreeze" later if we leave it in serial

    loss = custom_loss # custom_loss or 'mse'
    model.compile(loss=loss, optimizer=opt)

    model.summary()

    return model


def unfreeze_model(model, X, Y, freeze_fac=0.0, parallel=True):
    print("Unfreezing Model: make a new identical model, then copy the layer weights.")
    new_model = instantiate_model(X, Y, freeze_fac=freeze_fac)  # identical spec as original model, just not setting 'trainable=False'

    new_model.set_weights( make_serial( model, parallel=parallel).get_weights() )     # copy layer weights

    if parallel:
        new_model = make_parallel(new_model)

    opt = Adam()#lr=0.0001)
    loss = custom_loss # custom_loss or 'mse'
    new_model.compile(loss=loss, optimizer=opt)
    print("  ...finished un-freezing model")
    return new_model


# Loss functions

lambda_center = 1.0
lambda_size = 1.0
lambda_angle = 30.0
lambda_noobj = 0.3
lambda_class = 10.0

def custom_loss(y_true, y_pred):  # it's just MSE but the angle term is weighted by (a-b)^2
    print("custom_loss function engaged!")
    sqerr = K.square(y_true - y_pred)   # loss is 'built on' squared error
    pobj = 1 - y_true[:, ind_noobj::vars_per_pred]   # probability of object", i.e. existence.  if no object, then we don't care about the rest of the variables

    loss = lambda_center * ( K.sum(pobj* sqerr[:,ind_cx::vars_per_pred],     axis=-1) + K.sum(pobj* sqerr[:,ind_cy:-1:vars_per_pred], axis=-1))
    loss += lambda_size  * ( K.sum(pobj* sqerr[:,ind_semi_a::vars_per_pred], axis=-1) + K.sum(pobj* sqerr[:,ind_semi_b:-1:vars_per_pred], axis=-1))
    abdiff = y_true[:, ind_semi_a::vars_per_pred] - y_true[:, ind_semi_b:-1:vars_per_pred]
    loss += lambda_angle * K.sum(pobj* sqerr[:, ind_angle::vars_per_pred] * K.square(abdiff) , axis=-1)
    loss += lambda_noobj * K.sum(sqerr[:, ind_noobj::vars_per_pred], axis=-1)
    loss += lambda_class * K.sum( pobj * sqerr[:, ind_rings::vars_per_pred], axis=-1)

    # take average
    ncols = K.int_shape(y_pred)[-1]
    loss /= ncols
    return K.mean(loss)


def my_loss(y_true, y_pred):  # same as custom_loss but via numpy for easier diagnostics; inputs should be numpy arrays, not Tensors
    y_shape = y_pred.shape
    print("    my_loss: y_shape = ",y_shape)
    sqerr = (y_true - y_pred)**2   # loss is 'built on' squared error
    pobj = 1 - y_true[:, ind_noobj::vars_per_pred]   # probability of object", i.e. existence.  if no object, then we don't care about the rest of the variables

    center_loss = lambda_center * (np.sum(pobj* sqerr[:,ind_cx::vars_per_pred],     axis=-1) + np.sum(pobj* sqerr[:,ind_cy::vars_per_pred], axis=-1))
    size_loss   = lambda_size  * ( np.sum(pobj* sqerr[:,ind_semi_a::vars_per_pred], axis=-1) + np.sum(pobj* sqerr[:,ind_semi_b::vars_per_pred], axis=-1))
    abdiff = y_true[:, ind_semi_a::vars_per_pred] - y_true[:, ind_semi_b::vars_per_pred]
    angle_loss  = lambda_angle * np.sum(pobj * sqerr[:, ind_angle::vars_per_pred] * (abdiff**2) , axis=-1)
    noobj_loss  = lambda_noobj * np.sum(sqerr[:, ind_noobj::vars_per_pred], axis=-1)
    class_loss  = lambda_class * np.sum(pobj * sqerr[:, ind_rings::vars_per_pred], axis=-1)

    losses = np.array([center_loss, size_loss, angle_loss, noobj_loss, class_loss])
    losses = np.mean(losses,axis=-1)
    big_ind = np.argmax(losses)

    print("    my_loss: by class: [   center,        size,          angle,           noobj,          class]")
    print("              losses =",losses,", ind of biggest =",big_ind)
    loss = np.sum(losses)
    # take average
    ncols = y_pred.shape[-1]
    loss /= ncols
    return loss
