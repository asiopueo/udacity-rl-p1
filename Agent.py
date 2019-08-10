from collections import deque
import random

import Network

#
# Available actions are:
# 1. forward & backward
# 2. left & right
#

#
# State is given by 7 tuples of the form [1,2,3,4,5]:
# Each tuple describes a ray
# The 7 rays are emanated along the angles: [20,90,160,45,135,70,110]
# Or in ordered sequence: [20,45,70,90,110,135,160]
# [Yellow Banana, Wall, Blue Banana, Agent, Distance]
# Ex. [0,0,1,1,0,0.34] means there is
# The last 2 numbers are the left/right velocity v_yaw and the forward/backward velocity v_lat of the agent: [v_yaw, v_lat]



BUFFER_SIZE = 1000
actions = ['forward', 'backward', 'left', 'right']


class Agent():
    def __init__(self):
        #self.memory = ReplayBuffer()

        # QNetwork - We chose the simple network
        self.local_net = Network.network_simple()
        self.target_net = Network.network_simple()

    # Let the agent learn from experience
    def learn(self):
        pass

    # Take action - epsilon-greedy
    def action(self, state, epsilon=0.9):

        '''if random.random() > epsilon:
            return random.choice(actions)
        else:
            return self.local_net(state)
        '''

        return random.randrange(0,4)

    # Copy weights from short-term model to long-term model
    def update_long(self):
        self.long_net.set_weights( self.short_net.set_weights() )






class ReplayBuffer():
    def __init__(self):
        self.memory = deque(maxlen=BUFFER_SIZE)

    # Insert experience into memory
    def insert(self, experience):
        self.memory.append(experience)

    # Randomly sample memory
    def sample(self):
        state = buffer[x]
        action = buffer[x]
        reward = buffer[x]
        state_next = buffer[x]
        dome = buffer[x]

        return (state, action, reward, state_next, done)

    # Get length of memory
    def length(self):
        return len(self.memory)
