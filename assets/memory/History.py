#!/usr/bin/env python3

### normal python modules
import numpy as np
import torch

class History(object):
  '''
 stacks frames as some kind of histor for the agent since training data is
  not i.i.d
  '''
  def __init__(self, HIS_LENGTH):
    # self.history = np.zeros((HIS_LENGTH, 84, 84))
    self.history = np.zeros((1,HIS_LENGTH,84,84))
    self.capacity = HIS_LENGTH

  def __getitem__(self, key):
    return self.history[key]

  def stack(self, state):
    tmp_state = state.unsqueeze(1).numpy()
    self.history = np.delete(self.history,0,1)
    self.history = np.append(self.history,tmp_state,1)

    return torch.tensor(self.history,dtype=torch.float)
