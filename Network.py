import numpy as np

from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, BatchNormalization
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import adam
from keras.utils import np_utils, to_categorical # ???




def network_simple():
    model = Sequential()
    # The imput layer consists of 37 neurons (35 rays + 2 velocity)
    # The angles are emanated as follows:
    # [,,,,,,]
    model.add( Dense(37) )
    model.add( Dense(64) )
    model.add( Dense(64) )
    model.add( Dense(4) )

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def network_pixel():
    model = Sequential()

    model.add( Convolution2D(32, (3,3), input_shape=(), padding='valid', strides=(1,1), activation='relu', data_format='channels_first' )
    model.add( BatchNormalization() )
    model.add( MaxPooling2D( pool_size=(2,2), data_format='channels_first' ) )
    model.add( Convolution2D(64, (3,3), input_shape=(), padding='valid', strides=(1,1), activation='relu', data_format='channels_first' )
    model.add( BatchNormalization() )
    model.add( MaxPooling2D( pool_size=(2,2), data_format='channels_first') )

    model.add( Flatten() )

    model.add( Dense() )
    model.add( Dense() )
    model.add( Dense(4) )

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
