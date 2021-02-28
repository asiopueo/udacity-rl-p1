from collections import deque
import random
import os
import numpy as np
from collections import namedtuple

import Network


#import keras.backend as K

# Available actions are:
# 1. forward & backward
# 2. left & right

# State is given by 7 tuples of the form [1,2,3,4,5] and two additional scalars (cf. below):
# Each of the tuples describes a ray which has been emanated along the angles: [20,90,160,45,135,70,110]
# Or in ordered sequence: [20,45,70,90,110,135,160]
# [Yellow Banana, Wall, Blue Banana, Agent, Distance]
# Ex. [0,0,1,1,0,0.34] means there is
# The last 2 numbers are the left/right turning velocity v_yaw and the forward/backward velocity v_lat of the agent: [v_yaw, v_lat]


class DQNAgent():
    def __init__(self, buffer_size, batch_size, action_size, gamma):
        if not batch_size < buffer_size:
            raise Exception()

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_size = 4
        self.gamma = gamma

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        # Seed the random number generator
        random.seed()
        # QNetwork - We choose the simple network
        self.local_net = Network.network_simple()
        self.target_net = Network.network_simple()

    # Let the agent learn from experience
    def learn(self):
        # Retrieve batch of experiences from the replay buffer:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_from_buffer()
        # Prepare the target. Note that Q_target[:,action] will need to be assigned the 'true' learning target.
        #Q_target = np.zeros((self.batch_size, 4))
        td_targets = self.local_net.predict( state_batch )
        Q_target_a_max = np.max( self.target_net.predict(next_state_batch), axis=1 )

        states = []

        # Batches need to be prepared before learning
        for index, state in enumerate(state_batch):    
            # Calculate the next q-value according to SARSA-MAX   
            # Q_target w.r.t. action:
            if not done_batch[index]:
                td_targets[index, action_batch[index]] = reward_batch[index] + self.gamma * Q_target_a_max[index]
            else:
                td_targets[index, action_batch[index]] = reward_batch[index]

            states.append(state)

        self.local_net.fit(np.array( states ), np.array( td_targets ), batch_size=self.batch_size, epochs=1, shuffle=False, verbose=0)

    # Take action according to epsilon-greedy-policy:
    def action(self, state, epsilon=0.99):
        if random.random() > epsilon:
            return random.randrange(0, self.action_size)
        else:
            prob_distribution = self.local_net.predict(state.reshape(1,-1))
            action = np.argmax(prob_distribution)
            return action

    def random_action(self):
        return random.randrange(0, self.action_size)
        #return np.random.randint(self.action_size)

    # Copy weights from short-term model to long-term model (soft update)
    def update_target_net(self, tau=0.1):
        # Implement soft update for later:
        # get_weights()[0] -- weights
        # get weights()[1] -- bias (if existent)
        # Soft-update:
        local_weights = np.array( self.local_net.get_weights() )
        target_weights = np.array( self.target_net.get_weights() )
        self.target_net.set_weights( tau*local_weights + (1-tau)*target_weights )

    def load_weights(self, path):
        filepath = os.path.join(path, "dqn_weights_latest.ckpt")
        print("Loading network weights from", filepath)
        self.local_net.load_weights(filepath)
        self.target_net.load_weights(filepath)

    def save_weights(self, path):
        filepath = os.path.join(path, "dqn_weights_latest.ckpt")
        print("Saving target network weights to", filepath)
        self.target_net.save_weights(filepath)



class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=self.buffer_size)

    # Insert experience into memory
    def insert_into_buffer(self, experience):
        self.replay_buffer.append(experience)

    # Randomly sample memory
    def sample_from_buffer(self):
        # Sample experience batch from experience buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Reorder experience batch such that we have a batch of states, a batch of actions, a batch of rewards, etc.
        # Eventually add 'if exp is not None'
        state = np.vstack( [exp.state for exp in batch] )
        action = np.vstack( [exp.action for exp in batch] )
        reward = np.vstack( [exp.reward for exp in batch] )
        state_next = np.vstack( [exp.next_state for exp in batch] )
        done = np.vstack( [exp.done for exp in batch] )

        return state, action, reward, state_next, done

    # Get length of memory
    def buffer_usage(self):
        return len(self.replay_buffer) > self.batch_size
