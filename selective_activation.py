
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def selective_activation(x, start=1, end=-1, skip_every=6):
    new = K.identity(x)
    new[:,start:end:skip_every] = K.sigmoid(x[:,start:end:skip_every])
    return new

get_custom_objects().update({'selective_activation': Activation(selective_activation)})

model.add(Activation(selective_activation))
