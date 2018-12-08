

# Originally from https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
# SHHawley updated to Keras 2, added get_serial_part & get_available_gpus, and gpu_count=-1 flag

from keras.layers import concatenate
from keras.layers.core import Lambda
import keras.backend as K
from keras.models import Model
from keras.callbacks import Callback
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_serial_part(model, parallel=True):
    """
    Undoes make_parallel, but keyword included in case it's called on a serial model
    TODO: even better would be an internal check to see if model is parallel or serial
    """
    if (parallel):
        return model.layers[-2]
    else:
        return model                    # if model's already serial, return original model


def get_available_gpus():
    """
    from https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
      but updated thanks to http://blog.datumbox.com/5-tips-for-multi-gpu-training-with-keras/
    """
    local_device_protos = K.get_session().list_devices() #  device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def make_parallel(model):
    """
    Taken from https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
    Uses all available GPUs.  To limit number of GPUs used, call program from command line with CUDA_VISIBLE_DEVICES=...
    """
    gpu_list =  get_available_gpus()
    gpus_detected = len(gpu_list)
    if (gpus_detected < 2):
        return model    # no GPU parallelism

    gpu_count = gpus_detected

    print("make_parallel:",gpus_detected,"GPUs detected.  Parallelizing across",gpu_count,"GPUs...")

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            #merged.append(merge(outputs, mode='concat', concat_axis=0))
            merged.append(concatenate(outputs, axis=0))

        return Model(inputs=model.inputs, outputs=merged)
