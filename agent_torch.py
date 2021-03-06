from collections import deque
import random
import os
import numpy as np
from collections import namedtuple

from network_torch import QNetwork

from abc import ABC, abstractmethod
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cudo:0" if torch.cuda.is_available() else "cpu")

# Available actions are:
# 1. forward & backward
# 2. left & right

# State is given by 7 tuples of the form [1,2,3,4,5] and two additional scalars (cf. below):
# Each of the tuples describes a ray which has been emanated along the angles: [20,90,160,45,135,70,110]
# Or in ordered sequence: [20,45,70,90,110,135,160]
# [Yellow Banana, Wall, Blue Banana, Agent, Distance]
# Ex. [0,0,1,1,0,0.34] means there is
# The last 2 numbers are the left/right turning velocity v_yaw and the forward/backward velocity v_lat of the agent: [v_yaw, v_lat]


class TorchAgent():
    def __init__(self, buffer_size, batch_size, action_size, gamma, LR=0.0005):
        if not batch_size < buffer_size:
            raise Exception()

        self.state_size = 37
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_size = 4
        self.gamma = gamma
        self.lr = LR

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        # Seed the random number generator
        seed = 0#random.seed()
        # QNetwork - We choose the simple network
        self.local_net = QNetwork(self.state_size, self.action_size, seed)
        self.target_net = QNetwork(self.state_size, self.action_size, seed)
        self.optimizer = optim.Adam(self.local_net.parameters(), lr=self.lr)

        #self.hard_update_target_net()

    # Let the agent learn from experience
    def learn(self):
        # Check if buffer is sufficiently full:
        if not self.replay_buffer.buffer_usage():
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_from_buffer()

        qs_local = self.local_net.forward(state_batch)
        qsa_local = qs_local[torch.arange(self.batch_size, dtype=torch.long), action_batch.reshape(self.batch_size)]
        qsa_local = qsa_local.reshape((self.batch_size, 1))

        qs_target = self.target_net.forward(next_state_batch)
        _, qsa_local_argmax_a = torch.max(qs_local, dim=1) #using the greedy policy (q-learning)
        qsa_target = qs_target[torch.arange(self.batch_size, dtype=torch.long), qsa_local_argmax_a.reshape(self.batch_size)]

        qsa_target = qsa_target * (1 - done_batch.reshape(self.batch_size)) #target qsa value is zero when episode is complete
        qsa_target = qsa_target.reshape((self.batch_size,1))
        TD_target = reward_batch + self.gamma * qsa_target





    # Take action according to epsilon-greedy-policy:
    def action(self, state, epsilon=0.02):
        if random.random() < epsilon:
            return random.randrange(0, self.action_size)
        else:
            return random.randrange(0, self.action_size)

    def random_action(self):
        return random.randrange(0, self.action_size)
        #return np.random.randint(self.action_size)

    # Copy weights from short-term model to long-term model (soft update)
    def soft_update_target_net(self, tau=0.001):
        #local_weights = np.array( self.local_net.get_weights(), dtype=object )
        #target_weights = np.array( self.target_net.get_weights(), dtype=object )
        #self.target_net.set_weights( tau*local_weights + (1-tau)*target_weights )
        for t, l in zip(self.target_net.trainable_weights, self.local_net.trainable_weights):
            t.assign( (1-tau)*t + tau*l )

    def hard_update_target_net(self):
        self.target_net.set_weights( self.local_net.get_weights() )

    def load_weights(self, path):
        filepath = os.path.join(path, "ddqn_weights_latest.tf")
        print("Loading network weights from", filepath)
        self.local_net.load_weights(filepath)
        self.hard_update_target_net()

    def save_weights(self, path):
        filepath = os.path.join(path, "ddqn_weights_latest.tf")
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
        state = torch.from_numpy( np.vstack( [exp.state for exp in batch if exp is not None] )).float().to(device)
        action = torch.from_numpy( np.vstack( [exp.action for exp in batch if exp is not None] )).long().to(device)
        reward = torch.from_numpy( np.vstack( [exp.reward for exp in batch if exp is not None] )).float().to(device)
        state_next = torch.from_numpy( np.vstack( [exp.next_state for exp in batch if exp is not None] )).float().to(device)
        done = torch.from_numpy( np.vstack( [exp.done for exp in batch if exp is not None] )).float().to(device)

        return state, action, reward, state_next, done

    # Get length of memory
    def buffer_usage(self):
        return len(self.replay_buffer) > self.batch_size
