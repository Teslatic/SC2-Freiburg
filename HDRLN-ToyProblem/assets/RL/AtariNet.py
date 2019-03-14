#!/usr/bin/env python3

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# assign operations to GPU if possible
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

'''
AtariNet accordingly to the SC2-paper
Inputs:
    - non-spatial features
    - screen
    - minimap

- The input to the NN consists of an image with map dimensions (84,64)
- The first hidden layer convolves 32 filters of kernel size 5 with stride 4
- The activation function of the first layer is ReLU.
. The second hidden layer convolves 64 filters of kernel size 3 with stride 1
- The activation function of the second layer is ReLU.
- The channels are flatte out.
- The third hidden layer is fully-connected and consists of 512 rectifier units
- The output layer is fully-connected and the number of valid actions varies
depending on the minigame
and the agent.
'''


class DQN(nn.Module):

    def __init__(self, input_dim):
        super(DQN, self).__init__()
        self.num_actions = 3
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, self.num_actions)

        self.fc4 = nn.Linear(512, self.num_actions)
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, self.num_actions)

        self.dropout = nn.Dropout(p=0.5)

        self.ln1 = nn.LayerNorm(32)


        self.softmax = nn.Softmax(dim=-1)

    def _get_filter_dimension(self, w, f, p, s):
        '''
        calculates filter dimension according to following formula:
        (filter - width + 2*padding) / stride + 1
        '''
        return int((w - f + 2*p) / s + 1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # x = x.view(-1, self.num_flat_features(x))
        # print('x.shape', x.shape)
        # exit()
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        output = self.fc3(x)

        # x = F.relu(self.fc4(x))
        # x = self.dropout(F.relu(self.fc5(x)))
        # x = self.fc6(x)
        output = self.softmax(output)
        # print("output {}".format(output))

        return output
