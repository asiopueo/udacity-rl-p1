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
    fc1 = layers.Dense(64, kernel_initializer='glorot_uniform', activation='relu' )( input )
    fc2 = layers.Dense(64, kernel_initializer='glorot_uniform', activation='relu' )( fc1 )
    output = layers.Dense(4, kernel_initializer='glorot_uniform') ( fc2 )
    return keras.Model(inputs=input, outputs=output)

"""
def network_simple():
    model = keras.models.Sequential()
    model.add( layers.Dense(256, input_shape=(37,), kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu' ) )
    model.add( layers.Dense(128, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu' ) )
    model.add( layers.Dense(64, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu' ) )
    model.add( layers.Dense(4, kernel_initializer='glorot_uniform' , bias_initializer='zeros') )
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.002) )
    return model
"""

