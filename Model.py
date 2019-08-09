import QNetwork



# Available actions are:
# 1. forward & backward
# 2. left & right
#

# State is given by 7 tuples of the form [1,2,3,4,5]:
# Each tuple describes a ray
# The 7 rays are emanated along the angles: [10,,,90,,,170]
# The last 2 numbers are the speeds v_x and v_y of the agent
#


class Model():
    def __init__():
        self.memory = ReplayBuffer()

        # QNetwork - We chose the simple network
        self.local_net = Network.network_simple()
        self.target_net = Network.network_simple()

    # Let the agent learn from experience
    def learn():
        pass

    # Take action - epsilon-greedy
    def action(state, epsilon=0.9):

        if random > epsilon:
            return random(actions)
        else:
            return self.local_net()

    # Transfer
    def update_long():
        self.long_net = self.short_net



class ReplayBuffer():
    def __init()__:
        self.memory = dequeue()

    # Insert experience into memory
    def insert():
        pass

    # Randomly sample memory
    def sample():
        pass

    # Get length of memory
    def length():
        return
