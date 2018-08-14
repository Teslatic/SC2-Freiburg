#!/usr/bin/env python3

# normal python stuff
import random
import math
import numpy as np
import pandas as pd
from collections import namedtuple
from itertools import count

### pysc2 imports
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features


### torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

### assign operations to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
AtariNet accordingly to the SC2-paper
Inputs:
    - non-spatial features
    - screen
    - minimap
'''


# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'step_type'))

class ReplayBuffer(object):
    ''' Experience Replay Buffer Class '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'step_type'))

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        ''' store transition in replay buffer '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position +1) % self.capacity

    def sample(self, batch_size):
        ''' return random sample from ER '''
        return random.sample(self.memory, batch_size)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.num_actions = 2
        self.screen_dimension = 84

        # screen conv layers
        self.screen_conv1 = nn.Conv2d(1, 16, kernel_size=8, padding=0, stride =4)
        self.screen_conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=0, stride=2)

        # minimap conv layers
        self.minimap_conv1 = nn.Conv2d(3, 16, kernel_size=8, padding=0, stride =4)
        self.minimap_conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=0, stride=2)

        # fully connected layers
        self.screen_fc1 = nn.Linear(32*9*9,256)

        # action policy output
        self.action_fc1 = nn.Linear(256,self.num_actions)

        # x coordinate output
        self.x_coord_fc1 = nn.Linear(256,self.screen_dimension)

        # y coordinate output
        self.y_coord_fc1 = nn.Linear(256,self.screen_dimension)


    def forward(self, screen):
        screen = torch.from_numpy(screen).float()
        screen = F.relu(self.screen_conv1(screen))
        screen = F.relu(self.screen_conv2(screen))
        screen = screen.view(-1, 32*9*9)
        screen = F.relu(self.screen_fc1(screen))

        # estimated action q values
        action_q_values = self.action_fc1(screen.view(screen.size(0), -1))

        # estimated x coordinates q values
        x_coord_q_values = self.x_coord_fc1(screen.view(screen.size(0), -1))

        # estimated y coordinates q values
        y_coord_q_values = self.x_coord_fc1(screen.view(screen.size(0), -1))


        return action_q_values, x_coord_q_values, y_coord_q_values




if __name__ == '__main__':
    AtariNet = AtariNet()
