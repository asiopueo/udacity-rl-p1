import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, units_fc1 = 64, units_fc2 = 64):
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, units_fc1)
        self.fc2 = nn.Linear(units_fc1, units_fc2)
        self.fc3 = nn.Linear(units_fc2, action_size)

    def forward(self, state):
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )
        action = self.fc3( x )
        return action