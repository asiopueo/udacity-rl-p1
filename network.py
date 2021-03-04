import os
import numpy as np
os.environ['KMP_WARNINGS'] = 'FALSE'

import tensorflow as tf
from tensorflow import keras
from keras import layers

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("Using TensorFlow version", tf.version.VERSION)



# The input layer consists of 37 neurons (35 rays + 2 velocity)
def network_simple():
    input = layers.Input( shape=(37,) )
    fc1 = layers.Dense(64, kernel_initializer=tf.initializers.he_normal(), activation=keras.layers.ReLU() )( input )
    fc2 = layers.Dense(64, kernel_initializer=tf.initializers.he_normal(), activation=keras.layers.ReLU() )( fc1 )
    output = layers.Dense(4, kernel_initializer=tf.initializers.he_normal() ) ( fc2 )
    return keras.Model(inputs=input, outputs=output)

"""
def network_simple():
    model = keras.models.Sequential()
    model.add( layers.Dense(64, input_shape=(37,), kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu' ) )
    model.add( layers.Dense(64, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu' ) )
    model.add( layers.Dense(4, kernel_initializer='glorot_uniform' , bias_initializer='zeros') )
    model.compile(loss='mse', optimizer='adam' )
    return model
"""





# The pixel model for later research
"""
def model_pixel():
    model = Sequential()

    model.add( Convolution2D(32, (3,3), input_shape=(), padding='valid', strides=(1,1), activation='relu', data_format='channels_first' ))
    model.add( BatchNormalization() )
    model.add( MaxPooling2D( pool_size=(2,2), data_format='channels_first' ) )
    model.add( Convolution2D(64, (3,3), input_shape=(), padding='valid', strides=(1,1), activation='relu', data_format='channels_first' ))
    model.add( BatchNormalization() )
    model.add( MaxPooling2D( pool_size=(2,2), data_format='channels_first') )

    model.add( Flatten() )

    model.add( Dense() )
    model.add( Dense() )
    model.add( Dense(4) )

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
"""