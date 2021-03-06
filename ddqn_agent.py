import numpy as np
from abstract_agent import AbstractAgent

import tensorflow as tf
from tensorflow import keras

# Available actions are:
# 1. forward & backward
# 2. left & right

# State is given by 7 tuples of the form [1,2,3,4,5] and two additional scalars (cf. below):
# Each of the tuples describes a ray which has been emanated along the angles: [20,90,160,45,135,70,110]
# Or in ordered sequence: [20,45,70,90,110,135,160]
# [Yellow Banana, Wall, Blue Banana, Agent, Distance]
# Ex. [0,0,1,1,0,0.34] means there is
# The last 2 numbers are the left/right turning velocity v_yaw and the forward/backward velocity v_lat of the agent: [v_yaw, v_lat]

tau = 0.001

class DDQNAgent(AbstractAgent):
    def __init__(self, buffer_size, batch_size, action_size, gamma, algo_type='GRADIENT_TAPE'):
        super().__init__(buffer_size, batch_size, action_size, gamma)
        self.optimizer = tf.optimizers.Adam(lr=0.0005)
        self.algo_type = algo_type

    # Let the agent learn from experience
    #@tf.function
    def learn(self):
        # Check if buffer is sufficiently full:
        if not self.replay_buffer.buffer_usage():
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_from_buffer()

        if self.algo_type == 'COMPILE_FIT':
            # Keras-Compile-Fit-methodology:
            td_targets = self.local_net.predict( state_batch )
            Q_targets = self.target_net.predict( state_batch )
            a_max = np.argmax( self.local_net.predict(next_state_batch), axis=1 )

            for index, _ in enumerate(state_batch):    
                td_targets[index, action_batch[index]] = reward_batch[index] + self.gamma * Q_targets[index, a_max[index]] * (1-done_batch[index])

            #td_error = td_targets - self.local_net.predict( state_batch )
            #print(action_batch)
            #print(td_error)

            #self.local_net.fit(state_batch, td_targets, batch_size=self.batch_size, epochs=1, shuffle=False, verbose=0)
            loss = self.local_net.train_on_batch(state_batch, td_targets)
            #loss_ = tf.reduce_mean( tf.math.square(td_error))
            #print(loss)
            #print(loss_)

        elif self.algo_type == 'GRADIENT_TAPE':
            # GradientTape-methodology:
            state_batch = tf.convert_to_tensor(state_batch)

            """next_values = self.local_net([next_state_batch], training=False)
            a_max = tf.math.argmax(next_values, axis=1)
            Q_targets = self.target_net([state_batch], training=False)
            index_sequence = tf.stack( (tf.range(0, a_max.shape[0], dtype=tf.int64), a_max), axis=1 ) # Enumerated a_max-sequence
            Q_targets_max = tf.reshape(tf.gather_nd(Q_targets, index_sequence), (-1,1))
            td_targets = reward_batch + self.gamma * Q_targets_max * (1-done_batch)"""

            old_weights = self.local_net.trainable_weights

            with tf.GradientTape() as tape:
                #td_error = td_targets - self.local_net([state_batch], training=True)
                td_error = self.local_net([state_batch], training=True)
                loss = tf.math.reduce_mean( tf.math.square(td_error) ) 
        
            gradients = tape.gradient(loss, self.local_net.trainable_weights)
            keras.optimizers.Adam().apply_gradients( zip(gradients, self.local_net.trainable_weights ) )

            print(gradients[1])
            print("Before: ", old_weights[1])
            print("After: ", self.local_net.trainable_weights[1])


            #print(before==self.local_net.trainable_weights)

        self.soft_update_target_net( tau )





