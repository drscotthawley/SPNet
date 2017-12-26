#! /usr/bin/env python3
from __future__ import print_function

import numpy as np

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import InputLayer, Input, Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.applications import Xception, VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet, DepthwiseConv2D
from keras.utils.generic_utils import CustomObjectScope
from keras.applications.inception_v3 import preprocess_input
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
from keras.optimizers import SGD, Adam
from os.path import isfile
from utils import *
from multi_gpu import *



def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


def FullYOLO(X,Y): # after https://github.com/experiencor/basic-yolo-keras/blob/master/backend.py
                # see Table 6 in YOLO9000 paper, https://arxiv.org/pdf/1612.08242.pdf
    input_image = Input(shape=X[0].shape)

    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)

    def CBL_Block(inputs, nfeatures, kernel, num): # CBL = Conv/Batch/Leaky
        x = Conv2D(nfeatures, kernel, strides=(1,1), padding='same', name='conv_'+str(num), use_bias=False)(inputs)
        x = BatchNormalization(name='norm_'+str(num))(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    x = input_image
    x = CBL_Block(x,  32, (3,3),  1)                # 384 x 384
    x =    MaxPooling2D(pool_size=(2, 2))(x)        # 192 x 192
    x = CBL_Block(x,  64, (3,3),  2)
    x =    MaxPooling2D(pool_size=(2, 2))(x)        # 96 x 96

    x = CBL_Block(x, 128, (3,3),  3)
    x = CBL_Block(x,  64, (1,1),  4)
    x = CBL_Block(x, 128, (3,3),  5)
    x =    MaxPooling2D(pool_size=(2, 2))(x)        # 48 x 48

    x = CBL_Block(x, 256, (3,3),  6)
    x = CBL_Block(x, 128, (1,1),  7)
    x = CBL_Block(x, 256, (3,3),  8)
    x =    MaxPooling2D(pool_size=(2, 2))(x)       # 24 x 24

    x = CBL_Block(x, 512, (3,3),  9)
    x = CBL_Block(x, 256, (1,1), 10)
    x = CBL_Block(x, 512, (3,3), 11)
    x = CBL_Block(x, 256, (1,1), 12)
    x = CBL_Block(x, 512, (3,3), 13)
    x =    MaxPooling2D(pool_size=(2, 2))(x)       # 12 x 12

    skip_connection = x                            # um... I don't see this in the paper

    x = MaxPooling2D(pool_size=(2, 2))(x)          # 6 x 6

    x = CBL_Block(x,1024, (3,3), 14)
    x = CBL_Block(x, 512, (1,1), 15)
    x = CBL_Block(x,1024, (3,3), 16)
    x = CBL_Block(x, 512, (1,1), 17)
    x = CBL_Block(x,1024, (3,3), 18)               # Last layer in Table 6 of YOLO9000
    x = CBL_Block(x,1024, (3,3), 19)            # "We modify this network for detection by removing the last convolutional layer and instead
    x = CBL_Block(x,1024, (3,3), 20)            # adding on three 3 × 3 convolutional layers with 1024 filters each followed by a final
                                                # 1 × 1 convolutional layer with the number of outputs we need for detection.""


    skip_connection = CBL_Block(skip_connection, 64, (1,1), 21)  # Layer 21
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    #x = CBL_Block(x,1024, (3,3), 22)     # Layer 22
    x = CBL_Block(x, 16, (3,3), 22)     # my Layer 22
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    predictions = x

    model = Model(inputs=input_image, outputs=predictions)
    return model



def MyYOLONet(X,Y):
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    nfilters = 32

    nlayers1 = 4
    nlayers2 = 3
    inputs = Input(shape=X[0].shape)

    x = Conv2D(nfilters, kernel_size, strides=(1,1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    for n in range(nlayers1-1):
        x = Conv2D(nfilters, kernel_size, strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=pool_size)(x)

    skip_connection = x

    for n in range(nlayers2):
        x = Conv2D(nfilters*2, (1,1), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU(alpha=0.1)(x)

    skip_connection = Conv2D(32, (1,1), strides=(1,1), padding='same', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization()(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = DepthwiseConv2D((1,1))(skip_connection)
    #skip_connection = Lambda(space_to_depth_x2)(skip_connection)
    x = concatenate([skip_connection, x])

    x = Conv2D(nfilters, kernel_size, strides=(1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    x = Conv2D(16, (1,1), strides=(1,1), padding='same', use_bias=False)(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    x = Flatten()(x)
    predictions = x# #= Dense(Y[0].size,name='FinalOutput')(x)

    model = Model(inputs=inputs, outputs = predictions)
    return model


def instantiate_model(X, Y, freeze_fac=1.0):
    # Pick a pre-trained model, but leave the "top" off.

    #model = FullYOLO(X,Y)
    #return model

    input_tensor = Input(shape=X[0].shape)
    weights = None#'imagenet'    # None or 'imagenet'    If you want to use imagenet, X[0].shape[2] must = 3
    #base_model = VGG16(weights=weights, include_top=False, input_tensor=input_tensor)              # same for all inputs
    #--nope base_model = Xception(weights=weights, include_top=False, input_tensor=input_tensor)   # yield same #s for all inputs
    #base_model = InceptionV3(weights=weights, include_top=False, input_tensor=input_tensor)       # Works well, fast
    #base_model = InceptionResNetV2(weights=weights, include_top=False, input_tensor=input_tensor)  # works ok, big, slow.  doesn't play well with "unfreeze" in multi-gpu setting
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        base_model = MobileNet(input_shape=X[0].shape, weights=weights, include_top=False, input_tensor=input_tensor, dropout=0.6)       # Works well, VERY FAST! Needs img size 224x224
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
                loaded_model = make_serial( load_model(weights_file) , parallel=parallel)   # strip previous parallel part, to be added back in later

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
# Lambda values tuned by experience: to make corresponding losses close to the same magnitude
lambda_center = 1.0
lambda_size = 1.0
lambda_angle = 10.0
lambda_noobj = 0.5
lambda_class = 3.0

def custom_loss(y_true, y_pred):  # it's just MSE but the angle term is weighted by (a-b)^2
    print("custom_loss function engaged!")
    sqerr = K.square(y_true - y_pred)   # loss is 'built on' squared error
    pobj = 1 - y_true[:, ind_noobj::vars_per_pred]   # probability of object", i.e. existence.  if no object, then we don't care about the rest of the variables

    loss = lambda_center * ( K.sum(pobj* sqerr[:,ind_cx::vars_per_pred],     axis=-1) + K.sum(pobj* sqerr[:,ind_cy:-1:vars_per_pred], axis=-1))
    loss += lambda_size  * ( K.sum(pobj* sqerr[:,ind_semi_a::vars_per_pred], axis=-1) + K.sum(pobj* sqerr[:,ind_semi_b:-1:vars_per_pred], axis=-1))
    abdiff = y_true[:, ind_semi_a::vars_per_pred] - y_true[:, ind_semi_b:-1:vars_per_pred]
    loss += lambda_angle * ( K.sum(pobj* sqerr[:, ind_angle1::vars_per_pred] * K.square(abdiff) , axis=-1) + K.sum(pobj* sqerr[:, ind_angle2::vars_per_pred] * K.square(abdiff), axis=-1))
    loss += lambda_noobj * K.sum(sqerr[:, ind_noobj::vars_per_pred], axis=-1)
    loss += lambda_class * K.sum( pobj * sqerr[:, ind_rings::vars_per_pred], axis=-1)

    # take average
    ncols = K.int_shape(y_pred)[-1]
    loss /= ncols
    return K.mean(loss)


def my_loss(y_true, y_pred):  # diagnostic.  same as custom_loss but via numpy; inputs should be numpy arrays, not Tensors
    y_shape = y_pred.shape
    #print("    my_loss: y_shape = ",y_shape)
    sqerr = (y_true - y_pred)**2   # loss is 'built on' squared error
    pobj = 1 - y_true[:, ind_noobj::vars_per_pred]   # probability of object", i.e. existence.  if no object, then we don't care about the rest of the variables

    center_loss = lambda_center * (np.sum(pobj* sqerr[:,ind_cx::vars_per_pred],     axis=-1) + np.sum(pobj* sqerr[:,ind_cy::vars_per_pred], axis=-1))
    size_loss   = lambda_size  * ( np.sum(pobj* sqerr[:,ind_semi_a::vars_per_pred], axis=-1) + np.sum(pobj* sqerr[:,ind_semi_b::vars_per_pred], axis=-1))
    abdiff = y_true[:, ind_semi_a::vars_per_pred] - y_true[:, ind_semi_b::vars_per_pred]
    angle_loss  = lambda_angle * ( np.sum(pobj * sqerr[:, ind_angle1::vars_per_pred] * (abdiff**2) , axis=-1) + np.sum(pobj * sqerr[:, ind_angle2::vars_per_pred] * (abdiff**2) , axis=-1) )
    noobj_loss  = lambda_noobj * np.sum(sqerr[:, ind_noobj::vars_per_pred], axis=-1)
    class_loss  = lambda_class * np.sum(pobj * sqerr[:, ind_rings::vars_per_pred], axis=-1)

    losses = np.array([center_loss, size_loss, angle_loss, noobj_loss, class_loss])
    losses = np.mean(losses,axis=-1)

    ncols = y_pred.shape[-1]
    losses /= ncols            # take average
    ind_max = np.argmax(losses)

    print("    my_loss: [   center,        size,      angle,     noobj,      class ]")
    print("    losses =",losses,", ind_max =",ind_max)
    #print("              losses = [{:>8.4g}, {:>8.4g}, {:>8.4g}, {:>8.4g}, {:>8.4g}]".format(losses),", ind of biggest =",big_ind)
    loss = np.sum(losses)
    return loss, losses
