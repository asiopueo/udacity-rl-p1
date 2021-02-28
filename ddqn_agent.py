import numpy as np
from agent import AbstractAgent

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


class DDQNAgent(AbstractAgent):
    def __init__(self, buffer_size, batch_size, action_size, gamma):
        super().__init__(buffer_size, batch_size, action_size, gamma)

    # Let the agent learn from experience
    def learn(self):
        # Check if buffer is sufficiently full:
        if not self.replay_buffer.buffer_usage():
            return
        
        # Retrieve batch of experiences from the replay buffer:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_from_buffer()

        # Prepare the TD-target
        td_targets = self.local_net.predict( state_batch )
        Q_target = self.target_net.predict( state_batch )
        a_max = np.argmax( self.local_net.predict(next_state_batch), axis=1 )


        # Batches need to be prepared before learning
        for index, _ in enumerate(state_batch):    
            # Calculate the next q-value according to SARSA-MAX   
            # Q_target w.r.t. action:
            if not done_batch[index]:
                td_targets[index, action_batch[index]] = reward_batch[index] + self.gamma * Q_target[index, a_max[index]]
            else:
                td_targets[index, action_batch[index]] = reward_batch[index]

        #self.local_net.fit(state_batch, td_targets, batch_size=self.batch_size, epochs=1, shuffle=False, verbose=0)
        self.local_net.train_on_batch(state_batch, td_targets)
        self.update_target_net()





