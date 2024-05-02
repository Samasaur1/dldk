import torch
import numpy as np
from torch import nn, optim
import random

class DQN(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, num_actions)
        )

    def forward(self, x):
        return self.model(x)

    def act(self, x):
        return self(x).argmax()

class Q(nn.Module):
    # def __init__(self, input_shape, num_actions):
    #     super().__init__()
    #     self.input_shape = input_shape
    #     self.num_actions = num_actions
    #
    #     self.features = nn.Sequential(
    #         nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=8, stride=4),
    #         nn.ReLU(),
    #         nn.Conv2d(32, 64, kernel_size=4, padding=4, stride=2),
    #         nn.ReLU(),
    #         nn.Conv2d(64, 64, kernel_size=3, stride=1),
    #         nn.ReLU()
    #     )
    #     
    #     self.fc = nn.Sequential(
    #         nn.Linear(self.feature_size(), 512),
    #         nn.ReLU(),
    #         nn.Linear(512, self.num_actions)
    #     )
    #
    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     return x
    #
    # def feature_size(self):
    #     return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    def __init__(self, input_shape, num_actions):
        super().__init__()

        self.size = input_shape[0] * input_shape[1] * input_shape[2]

        self.model = nn.Sequential(
            nn.Linear(self.size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.model(x.reshape(self.size))


class ReplayMemory:

    def __init__(self, cap):
        self.capacity = cap
        self.data = []

    def push(self, state, action, reward, nstate, term):
        if len(self.data) < self.capacity:
            self.data.append((state, action, reward, nstate, term))
        else:
            idx = random.randint(0, self.capacity - 1)
            self.data[idx] = (state, action, reward, nstate, term)

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)


import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions import Categorical
class ActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        dist = Categorical(x)
        return dist
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
