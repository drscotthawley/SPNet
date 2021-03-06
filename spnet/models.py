#! /usr/bin/env python3
from __future__ import print_function

import numpy as np

import keras
import keras.backend as K
from keras.engine.topology import Layer
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Lambda, Concatenate, Add, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.applications import Xception, VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.densenet import DenseNet121

from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.layers import DepthwiseConv2D    # newer keras
from keras.applications.mobilenet import MobileNet, DepthwiseConv2D   # older keras

from keras.applications.inception_v3 import preprocess_input
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
from keras.optimizers import SGD, Adam

from keras.utils.generic_utils import get_custom_objects, CustomObjectScope
import tensorflow as tf
from os.path import isfile
from spnet import multi_gpu, utils
import spnet.config as cf
from functools import partial

from keras.utils.generic_utils import get_custom_objects
import tempfile
from keras import regularizers

import sys
import os
import tempfile


def str_to_class(str):  # Used to clean up definitions of models
    return getattr(sys.modules[__name__], str)

# from https://gist.github.com/sthalles/d4e0c4691dc2be2497ba1cdbfe3bc2eb#file-add_regularization-py
def add_regularization(model, regularizer=regularizers.l2(0.0001)):

    print("Adding regularization")
    #if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
    #  print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
    #  return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = keras.models.model_from_json(model_json, {'SelectiveSigmoid': SelectiveSigmoid})

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


class Mish(Activation):  # Mish is intereting but I get OOM errors.
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'
def mish(x, fast=False):
    if fast:              # faster but requires extra storage
        y = K.exp(-x)
        z = 1 + 2 * y
        return x * z / (z + 2* y * y)
    #return x * tf.math.tanh(tf.math.softplus(x))
    return x * K.tanh(K.softplus(x))

get_custom_objects().update({'Mish': Mish(mish)})




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


class InterleaveColumns(Layer):
    """
    This takes a concatted tensor where the first n_preds are interleaved with the remaining columns
    Output shape is the same as the input shape.  All this does is rearrange columns

    Example: interleave the [10,11,12] with the [1,2,3,4,5,6], starting at index 2
    input =
     [[10. 11. 12.  1.  2.  3.  4.  5.  6.]
     [10. 11. 12.  1.  2.  3.  4.  5.  6.]
     [10. 11. 12.  1.  2.  3.  4.  5.  6.]]
    result =
     [[ 1.  2. 10.  3.  4. 11.  5.  6. 12.]
     [ 1.  2. 10.  3.  4. 11.  5.  6. 12.]
     [ 1.  2. 10.  3.  4. 11.  5.  6. 12.]]

     This is used for integrating the "noobj" values (index=6?)
    """
    def make_perm_matrix(self, input_shape, start_index=6):
        n_vars = input_shape[-1]
        if (n_vars % vars_per_pred != 0):
            raise ValueError("n_vars (="+str(n_vars)+") must be a multiple of vars_per_pred (="+str(cf.vars_per_pred)+")")

        n_preds = int(n_vars / cf.vars_per_pred)
        # cml = "column map list" is the list of where each column will get mapped to
        cml = [start_index + x*(cf.vars_per_pred) for x in range(n_preds)]  # first array
        for i in range(n_preds):                                       # second array
            cml += [x + i*(cf.vars_per_pred) for x in range(start_index)] # vars before start_index
            cml += [1 + x + i*(cf.vars_per_pred) + start_index \
                for x in range(cf.vars_per_pred-start_index-1)]           # vars after start_index

        # Create a permutation matrix using cml
        np_perm_mat = np.zeros((len(cml), len(cml)))
        for idx, i in enumerate(cml):
            np_perm_mat[idx, i] = 1
        perm_mat = K.constant(np_perm_mat, dtype=cf.dtype) # see utils.py for dtype def
        return perm_mat

    def __init__(self, start_index=6, **kwargs):
        super(InterleaveColumns, self).__init__(**kwargs)
        self.start_index = start_index
        return # save this for build: self.perm_mat =  make_perm_matrix():

    def get_output_shape_for(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.perm_mat = self.make_perm_matrix(input_shape, start_index=self.start_index)
        super(InterleaveColumns, self).build(input_shape)
        return

    def call(self, x):
        return K.dot(x, self.perm_mat)   # here's where the reordering is applied


class SelectiveSigmoid(Layer):
    """
    Applies sigmoid activation to only certain strided variables in a layer,
    leaves the other values untouched (i.e. linear activation)
    """
    def __init__(self, **kwargs):
        self.start = kwargs.get('start', cf.ind_noobj)
        self.end = kwargs.get('end', None)
        self.skip = kwargs.get('skip', cf.vars_per_pred)
        self.sigmoid_stretch = 1                        # some ppl make the sigmoid>1  (e.g. LeCun et al, "Efficient Backprop", 1998), but this will break CE loss
        super(SelectiveSigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        self.indices = np.zeros(input_shape[-1])           # Note that tf.cast (below) allows a numpy array
        self.indices[self.start:self.end:self.skip] = 1    # since it's numpy, we can do sliced assignment

    def call(self, x):
        # in the following line, the transposing is necessary to obviate the need to know the batch_size
        return tf.transpose(tf.where(tf.cast(self.indices, dtype=tf.bool), self.sigmoid_stretch*K.sigmoid(tf.transpose(x)), tf.transpose(x)))

    def compute_output_shape(self, input_shape):
        return input_shape



def create_model_functional(X, Y0size=576, freeze_fac=0.75, quick_setup=False):
    """
    accepts a 'large' grayscale image (or set of images), runs a convnet & pooling to shrink it in a learnable way
    #  this produces a smaller set of 3 'color' images for different features.

    This newer model incorporates sigmoid activations for existence / noobj predictions, and uses an
      "InterleaveColumns" layer to reorder the columns and produce output compatible with the previous model.
    """
    print("Using functional API model, cf.basemodel =",cf.basemodel)

    # First part of the model is to take the large grayscale image, shrink it, and
    # add 3 conv operations to produce something akin to color channels to feed into the
    # 'standard' models below:
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    nfilters = 3        # adding 3 'color' channels
    print("X[0].shape = ",X[0].shape)
    inputs = Input(shape=X[0].shape)
    x = inputs
    x = Conv2D(nfilters, kernel_size, strides=(1,1), padding='same', use_bias=False)(x) # give us 3 colors

    x = AveragePooling2D(pool_size=pool_size)(x)                                       # shrink image

    # ResNet block:
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    #x  = Activation('Mish')(x)

    x = Conv2D(nfilters, kernel_size, strides=(1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    #x  = Activation('Mish')(x)

    x = Conv2D(nfilters, kernel_size, strides=(1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x =  Add()([x, AveragePooling2D(pool_size=pool_size)(inputs)])  # residual skip connection on shrunk input
    #x =  Add()([x, id])  # residual skip connection on shrunk input

    x = Dropout(0.1)(x)
    # Finished with first part of the model
    print("After initial convolutions & pooling, x.shape = ",x.shape)

    # 'PreFab'/standard CNN middle section
    #weights = 'imagenet'  # None or 'imagenet'.  Note: If you ever get "You are trying to load a model with __layers___", you need to add by_name=True in the load_weights call for your Prefab CNN
    weights = None
    weights = "imagenet"
    base_model_class = str_to_class(cf.basemodel)
    if cf.basemodel == 'MobileNet':
        # with CustomObjectScope({'relu6': ReLU(6.),'DepthwiseConv2D': DepthwiseConv2D}):  # newer keras
        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'custom_loss':custom_loss, 'tf':tf}):   # older keras
            input_shape = x.shape[1:4]
            print("input_shape = ",input_shape)
            print("input_shape[-1] = ",input_shape[-1])
            base_model = MobileNet(input_shape=input_shape, weights=weights, include_top=False, input_tensor=x, dropout=0.6)       # Works well, VERY FAST! Needs img size 224x224
    else:
        base_model_class = getattr(sys.modules[__name__], cf.basemodel )  # use basemodel string to load Keras model
        # Note: this requires modifying keras file applications/<model_name>.py to add ", by_name=True" in the load_weights() line(s)
        base_model = base_model_class(weights=None, include_top=False, input_tensor=x)

    num_layers = len(base_model.layers)
    freeze_layers = int(num_layers * freeze_fac)
    print("Freezing ",freeze_layers,"/",num_layers," layers of base_model")
    if (freeze_fac == 1.0):
        base_model.trainable = False
    else:
        base_model.trainable = True
    # set the first N layers of the base model (e.g., p to the last conv block)
    # to non-trainable (weights will not be updated)
    # for fine-tuning, "freeze" most of the pre-trained model, and then unfreeze it later
    if (freeze_layers > 0 ):
        for i in range(freeze_layers):
            base_model.layers[i].trainable = False

    print("base_model.output_shape =",base_model.output_shape)

    # Finally we stack on top, a 'flat' output
    x = Flatten(input_shape=base_model.output_shape[1:])(base_model.output)
    if cf.model_type == 'compound':   # note, this is rarely used, instead we use either SelectiveSigmoid or just alter the loss function
        if (Y0size % cf.vars_per_pred != 0):
            raise ValueError("Y0size (="+str(Y0size)+") must be a multiple of cf.vars_per_pred (="+str(cf.vars_per_pred)+")")
        n_preds = int(Y0size / cf.vars_per_pred)
        sigout = Dense(n_preds, activation='sigmoid', name='SigmoidOutput')(x) # activation='sigmoid',
        denseout = Dense(Y0size-n_preds, name='DenseOutput')(x)
        top = Concatenate()([sigout, denseout])
        top = InterleaveColumns(start_index=cf.ind_noobj, name='FinalOutput')(top)
    else:
        top = Dense(Y0size, name='FinalOutput')(x)

    ''' # We'll handle 'same' loss function option in the loss routine itself
    if cf.loss_type != 'same':  # cf.model_type == 'ss':
        print("**Adding SelectiveSigmoid**")
        top = SelectiveSigmoid(name='ReallyFinalOutput')(top) # make the existence variables sigmoids
    '''

    model = Model(inputs=inputs, outputs=top)

    if quick_setup:
        return model

    model = add_regularization(model)
    print("After adding l2 regularization, model.losses =",model.losses)

    # One more time just for good measure: Freeze base model
    # set the first N layers of the base model (e.g., p to the last conv block)
    # to non-trainable (weights will not be updated)
    # for fine-tuning, "freeze" most of the pre-trained model, and then unfreeze it later
    num_layers = len(base_model.layers)
    freeze_layers = int(num_layers * freeze_fac)
    if (freeze_layers > 0 ):
        print("again: Freezing ",freeze_layers,"/",num_layers," layers of base_model")
        for i in range(freeze_layers):
            base_model.layers[i].trainable = False

    # show how many trainable parameters there are
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    print('create_model_functional: Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('create_model_functional: Trainable params: {:,}'.format(trainable_count))
    print('create_model_functional: Non-trainable params: {:,}'.format(non_trainable_count))
    return model



def create_model_simple(X, Y0size=576, freeze_fac=0.75):
    """
    creates a model from scratch.  Y0size is the size of the (flattened) output grid of predictors, e.g. 6x6x2x8=576
    """

    # Simple model (rarely used anymore)
    #   Pick a pre-trained model, but leave the "top" off.  Note that for pretrained weights, these models generally
    #   require 3 color channels, and a small input size (e.g. 224x224)
    input_tensor = Input(shape=X[0].shape)
    weights = 'imagenet'    # None or 'imagenet'    If you want to use imagenet, X[0].shape[2] must = 3
    base_model = NASNetMobile(input_shape=X[0].shape, weights=weights, include_top=False, input_tensor=input_tensor)
    top_model = Sequential()        # top_model gets tacked on to pretrained model
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))   # This is the 'grid of predictors'.  It's actually a flat layer.
    top_model.add(Dense(Y0size, name='FinalOutput'))      # Final output layer
    if (cf.loss_type != 'same'):
        print("**Adding SelectiveSigmoid**")
        top_model.add(SelectiveSigmoid(name='ReallyFinalOutput')) # make the existence variables sigmoids
    model = Model(inputs= base_model.input, outputs= top_model(base_model.output))


    # Freezing: set the first N layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # for fine-tuning, "freeze" most of the pre-trained model, and then unfreeze it later
    num_layers = len(base_model.layers)
    freeze_layers = int(num_layers * freeze_fac)
    if (freeze_layers > 0 ):
        print("Freezing ",freeze_layers,"/",num_layers," layers of base_model")
        for i in range(freeze_layers):
            base_model.layers[i].trainable = False

    return model


def setup_model(X, Y0size=576, try_checkpoint=True, no_cp_fatal=False, \
    weights_file='weights.hdf5', freeze_fac=0.75, parallel=False, quick_setup=False):
    """
    this is the main routine for setting up the NN model, either 'from scratch' or loading from checkpoint
    """

    print("Initializing blank model: Y0size =",Y0size)
    if cf.model_type == 'simple':
        model = create_model_simple(X, Y0size=Y0size, freeze_fac=freeze_fac)
    else:
        model = create_model_functional(X, Y0size=Y0size, freeze_fac=freeze_fac, quick_setup=quick_setup)


    # Initialize weights using checkpoint if it exists.
    if (try_checkpoint):
        if ( isfile(weights_file) ):
            print ('Weights file detected. Loading from',weights_file)
            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'custom_loss':custom_loss, 'tf':tf}):  # old keras
            #with CustomObjectScope({'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D, 'custom_loss':custom_loss, 'tf':tf}): # new keras
                model.load_weights(weights_file)    # Note: assume serial part was saved, not parallel model
        else:
            if (no_cp_fatal):
                raise Exception("*** No weights file detected; can't do anything.  Aborting.")
            else:
                print('    No weights file detected, so starting from scratch.')

    serial_model = model


    #if quick_setup:
    #    print("setup_model: quick_setup=True, returning early")
    #    return model, serial_model

    opt = Adam(lr=0.00001)
    loss = custom_loss # custom_loss or 'mse'

    if parallel and (len(multi_gpu.get_available_gpus())>1):
        model = multi_gpu.make_parallel(model)    # Note: easier to "unfreeze" later if we leave it in serial

    #if not quick_setup:                    # workaround to issues loading weights
    print("Compiling the model")
    model.compile(loss=loss, optimizer=opt)

    #print("Model summary:")    # Model summary for pre-fab CNN models is way too long. Omit
    #model.summary()

    return model, serial_model


def unfreeze_model(model, X, Y, parallel=False):
    print("Unfreezing Model: make a new identical model, then copy the layer weights.")

    if cf.model_type == 'simple':
        new_model = create_model_simple(X, Y[0].size, freeze_fac=0)  # identical spec as original model, just not setting 'trainable=False'
    else:
        new_model = create_model_functional(X, Y[0].size, freeze_fac=0)  # identical spec as original model, just not setting 'trainable=False'

    new_model.set_weights( multi_gpu.get_serial_part( model, parallel=parallel).get_weights() )     # copy layer weights
    # Add regularizer
    """
    print("Adding l2 regularization")
    #  see https://sthalles.github.io/keras-regularizer/
    from keras import regularizers
    l2 = regularizers.l2(1e-4)
    for layer in new_model.layers:
    # if hasattr(layer, 'kernel'):
    # or
    # If you want to apply just on Conv
        if isinstance(layer, keras.layers.Conv2D):
            new_model.add_loss(lambda layer=layer: l2(layer.kernel))
    print("After l2 regularization, new_model.losses =",new_model.losses)
    """

    if parallel:
        new_model = multi_gpu.make_parallel(new_model)

    opt = Adam(lr=0.00001)
    loss = custom_loss # custom_loss or 'mse'
    new_model.compile(loss=loss, optimizer=opt)
    print("  ...finished un-freezing model")

    # show how many trainable parameters there are
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(new_model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(new_model.non_trainable_weights)]))

    print('post-unfreeze_model: Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('post-unfreeze_model: Trainable params: {:,}'.format(trainable_count))
    print('post-unfreeze_model: Non-trainable params: {:,}'.format(non_trainable_count))

    return new_model


# Loss functions
# Lambda values tuned by experience: to make corresponding losses close to the same magnitude
lambda_center = 2.0
lambda_size = 1.0
lambda_angle = 3.0
lambda_noobj = 0.3   # note that other parts of loss are scaled by *ground truth* noobj/p, not predicted noobj/p
lambda_class = 5.0
logeps = 1e-10       # small number to avoid log(0) errors

def custom_loss(y_true, y_pred):  # it's just MSE but the angle term is weighted by (a-b)^2
    print("custom_loss function engaged!. cf.loss_type =",cf.loss_type)
    sqerr = K.square(y_true - y_pred)   # loss is 'built on' squared error
    pobj_true = 1 - y_true[:, cf.ind_noobj::cf.vars_per_pred]   # "probability of object", i.e. existence.  if no object, then we don't care about the rest of the variables

    if cf.loss_type=='same':  # MSE everywhere
        loss = lambda_noobj * K.sum(sqerr[:, cf.ind_noobj::cf.vars_per_pred], axis=-1)  # MSE version
    else:
        # unstable/naive:
        #    pobj_pred = 1 - y_pred[:, cf.ind_noobj::cf.vars_per_pred]  # this variable is only used for BCE loss
        #    loss = lambda_noobj * (-K.sum(  pobj_true*K.log(pobj_pred+logeps) + (1-pobj_true)*tf.log1p(-pobj_pred) , axis=-1) )
        # stable/clever:
        noobj_true = y_true[:, cf.ind_noobj::cf.vars_per_pred]  # just defined for conveniece / readability
        z =          y_pred[:, cf.ind_noobj::cf.vars_per_pred]  # treat noobj pred values as logits (don't use SelectiveSigmoid!)
        loss  = lambda_noobj * K.sum( K.maximum(0.0, z) - z*noobj_true + tf.log1p(K.exp(-K.abs(z))), axis=-1 )

    loss += lambda_center * ( K.sum(pobj_true* sqerr[:,cf.ind_cx::cf.vars_per_pred],     axis=-1) + K.sum(pobj_true* sqerr[:,cf.ind_cy:-1:cf.vars_per_pred], axis=-1))
    loss += lambda_size  * ( K.sum(pobj_true* sqerr[:,cf.ind_semi_a::cf.vars_per_pred], axis=-1) + K.sum(pobj_true* sqerr[:,cf.ind_semi_b:-1:cf.vars_per_pred], axis=-1))
    abdiff = y_true[:, cf.ind_semi_a::cf.vars_per_pred] - y_true[:, cf.ind_semi_b:-1:cf.vars_per_pred]
    loss += lambda_angle * ( K.sum(pobj_true* sqerr[:, cf.ind_angle1::cf.vars_per_pred] * K.square(abdiff) , axis=-1) + K.sum(pobj_true* sqerr[:, cf.ind_angle2::cf.vars_per_pred] * K.square(abdiff), axis=-1))
    loss += lambda_class * K.sum( pobj_true * sqerr[:, cf.ind_rings::cf.vars_per_pred], axis=-1)

    # take average
    ncols = K.int_shape(y_pred)[-1]
    loss /= ncols
    return K.mean(loss)


# same as custom_loss but via numpy; inputs should be numpy arrays, not Tensors
#  And this gives more 'fine-grained' output for diagnostic purposes
def my_loss(y_true, y_pred, verbosity=0):  # diagnostic.
    y_shape = y_pred.shape

    sqerr = (y_true - y_pred)**2   # almost all of loss is 'built on' squared error

    pobj_true = 1 - y_true[:, cf.ind_noobj::cf.vars_per_pred] # =0's & 1's.  true "probability of object", i.e. existence.  if no object, then we don't care about the rest of the variables

    if cf.loss_type=='same':  # MSE everywhere, including noobj / pobj
        noobj_loss  = lambda_noobj * np.sum(sqerr[:, cf.ind_noobj::cf.vars_per_pred], axis=-1)  # MSE loss
    else:                     # BCE loss for existence
        # Naive/unstable way to do this:  (will yield NaNs eventually)
        #      pobj_pred = 1 - y_pred[:, cf.ind_noobj::cf.vars_per_pred]  # this variable is only used for BCE loss
        #      noobj_loss = lambda_noobj * (-np.sum(  pobj_true*np.log(pobj_pred+logeps) + (1-pobj_true)*np.log1p(-pobj_pred) , axis=-1) )  # Cross-entropy loss
        # Clever/stable way:
        noobj_true = y_true[:, cf.ind_noobj::cf.vars_per_pred]  # aux variable for convenience / readability
        z =          y_pred[:, cf.ind_noobj::cf.vars_per_pred]  # treat noobj pred values as logits (don't use SelectiveSigmoid!)
        noobj_loss  = lambda_noobj * np.sum( np.maximum(0.0, z) - z*noobj_true + np.log1p(np.exp(-np.abs(z))), axis=-1 )

    center_loss = lambda_center * (np.sum(pobj_true* sqerr[:,cf.ind_cx::cf.vars_per_pred],     axis=-1) + np.sum(pobj_true* sqerr[:,cf.ind_cy::cf.vars_per_pred], axis=-1))
    size_loss   = lambda_size  * ( np.sum(pobj_true* sqerr[:,cf.ind_semi_a::cf.vars_per_pred], axis=-1) + np.sum(pobj_true* sqerr[:,cf.ind_semi_b::cf.vars_per_pred], axis=-1))
    abdiff = y_true[:, cf.ind_semi_a::cf.vars_per_pred] - y_true[:, cf.ind_semi_b::cf.vars_per_pred]
    angle_loss  = lambda_angle * ( np.sum(pobj_true * sqerr[:, cf.ind_angle1::cf.vars_per_pred] * (abdiff**2) , axis=-1) + np.sum(pobj_true * sqerr[:, cf.ind_angle2::cf.vars_per_pred] * (abdiff**2) , axis=-1) )
    class_loss  = lambda_class * np.sum(pobj_true * sqerr[:, cf.ind_rings::cf.vars_per_pred], axis=-1)

    losses = np.array([center_loss, size_loss, angle_loss, noobj_loss, class_loss])
    losses = np.mean(losses,axis=-1)

    ncols = y_pred.shape[-1]
    losses /= ncols            # take average
    ind_max = np.argmax(losses)

    if verbosity>0:
        print("    my_loss: [   center,        size,      angle,     noobj,      class ].  loss_type =",cf.loss_type)
        print("    losses =",losses,", ind_max =",ind_max)
        print("pobj_pred is <= 0 at indices ",np.argwhere(pobj_pred <= 0),", = ",pobj[ np.argwhere(pobj_pred <= 0) ])
        print("1-pobj_pred is <= 0 at indices ",np.argwhere(1-pobj_pred <= 0),", = ",pobj[np.argwhere(1-pobj_pred <= 0)])


    loss = np.sum(losses)
    return loss, losses
