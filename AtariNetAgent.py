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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    ''' Experience Replay Buffer Class '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def store_transition(self, *args):
        ''' store transition in replay buffer '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position +1) % self.capacity

    def sample(self, batch_size):
        ''' return random sample from ER '''
        return random.sample(self.memory, batch_size)


class AtariNet(nn.Module):

    def __init__(self):
        super(AtariNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, padding=0, stride =4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=0, stride=2)











if __name__ == '__main__':
    AtariNet = AtariNet()
    print(AtariNet)
