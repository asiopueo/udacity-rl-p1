

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
        
