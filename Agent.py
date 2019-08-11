from collections import deque
import random
import numpy as np
from collections import namedtuples

import Network

import keras.backend as K

# Available actions are:
# 1. forward & backward
# 2. left & right

# State is given by 7 tuples of the form [1,2,3,4,5]:
# Each tuple describes a ray
# The 7 rays are emanated along the angles: [20,90,160,45,135,70,110]
# Or in ordered sequence: [20,45,70,90,110,135,160]
# [Yellow Banana, Wall, Blue Banana, Agent, Distance]
# Ex. [0,0,1,1,0,0.34] means there is
# The last 2 numbers are the left/right velocity v_yaw and the forward/backward velocity v_lat of the agent: [v_yaw, v_lat]


BUFFER_SIZE = 1000
BATCH_SIZE = 10
GAMMA = 0.98

class Agent():
    def __init__(self):
        # Initialize replay buffer
        self.memory = ReplayBuffer()

        # Seed the random number generator
        random.seed()

        # QNetwork - We choose the simple network
        self.local_net = Network.network_simple()
        self.target_net = Network.network_simple()


    # Let the agent learn from experience
    def learn(self):
        # Retrieve batch of experiences from the replay buffer
        batch = self.memory.sample_from_buffer()

        # Calculate the next q-value according to SARSA-MAX
        Q_next = np.argmax( self.target_net.predict(state.reshape(1,-1)) )
        # Target:
        Q_target = reward + GAMMA * Q_next

        # Error: Look into Network.py for choise of loss function
        Q_local = self.local_net.predict( state.reshape(1,-1) )

        # The loss function is given by
        self.local_net.fit(Q_local, Q_target)
        loss = self.local_net.evaluate()


    # Take action according to epsilon-greedy-policy:
    def action(self, state, epsilon=0.9):

        '''if random.random() > epsilon:
            return random.randrange(0,4)
        else:
            return self.local_net(state)
        '''
        prob_distribution = self.local_net.predict(state.reshape(1,-1))
        action = np.argmax(prob_distribution)
        return action

    # Copy weights from short-term model to long-term model
    def update_long_net(self, tau=1.0):
        # Implement soft update for later
        self.long_net.set_weights( self.short_net.set_weights() )







class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    # Insert experience into memory
    def insert_into_buffer(self, experience):
        self.buffer.append(experience)

    # Randomly sample memory
    def sample_from_buffer(self):
        # Sample experience batch from experience buffer
        batch = random.sample(self.buffer, BATCH_SIZE)

        # Reorder experience batch such that we have a bach of states, a batch of actions, a batch of rewards, etc.
        # Eventually add 'if exp is not None'
        state = np.vstack( [exp.state for exp in batch] )
        action = np.vstack( [exp.action for exp in batch] )
        reward = np.vstack( [exp.reward for exp in batch] )
        state_next = np.vstack( [exp.state_next for exp in batch] )
        done = np.vstack( [exp.done for exp in batch] )

        return (state, action, reward, state_next, done)

    # Get length of memory
    def buffer_usage(self):
        return len(self.buffer)
