import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layers = [64,64]):
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layers[1], action_size)

    def forward(self, state):
        state = self.fc1(state)
        state = self.relu1(state)
        state = self.fc2(state)
        state = self.relu2(state)
        state = self.fc3(state)
        return state