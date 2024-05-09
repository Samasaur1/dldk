import torch
import numpy as np
from torch import nn, optim
import random

class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(352, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 352)
        x = self.fc(x)
        return x

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
