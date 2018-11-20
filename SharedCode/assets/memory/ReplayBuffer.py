#!/usr/bin/env python3

import random
from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayBuffer(object):
    ''' Experience Replay Buffer Class '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, key):
        return self.memory[key]

    def push(self, *args):
        ''' store transition in replay buffer '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        ''' return random sample transition from ER '''
        return random.sample(self.memory, batch_size)
