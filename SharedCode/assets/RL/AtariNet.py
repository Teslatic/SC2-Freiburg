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

    def __init__(self, history_length, dim_actions):
        super(DQN, self).__init__()
        self.num_actions = dim_actions
        self.map_dimensions = (84, 64)
        self.history_length = history_length

        # screen conv layers
        self.screen_conv1 = nn.Conv2d(in_channels=1,
                                      out_channels=16,
                                      kernel_size=5,
                                      padding=0,
                                      stride=4)
        self.screen_conv2 = nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=3,
                                      padding=0,
                                      stride=1)
        # self.screen_conv3 = nn.Conv2d(in_channels=32
        #                                 out_channels=64,
        #                                 kernel_size=3,
        #                                 stride=1)

        # fully connected layers
        self.tmp_w = self._get_filter_dimension(84, 5, 0, 4)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, 3, 0, 1)
        # self.tmp_w = self._get_filter_dimension(self.tmp_w, 3, 0, 1)
        self.screen_fc1 = nn.Linear(32*self.tmp_w*self.tmp_w, 512)
        self.action_fc1 = nn.Linear(512, self.num_actions)

    def _get_filter_dimension(self, w, f, p, s):
        '''
        calculates filter dimension according to following formula:
        (filter - width + 2*padding) / stride + 1
        '''
        return int((w - f + 2*p) / s + 1)

    def forward(self, screen):
        screen = F.relu(self.screen_conv1(screen))
        screen = F.relu(self.screen_conv2(screen))
        # screen = F.relu(self.screen_conv3(screen))
        screen = screen.view(-1, 32*self.tmp_w*self.tmp_w)
        screen = F.relu(self.screen_fc1(screen))
        action_q_values = self.action_fc1(screen)
        return action_q_values


class SingleDQN(nn.Module):

    def __init__(self, history_length, num_outputs):
        super(SingleDQN, self).__init__()
        self.num_outputs = num_outputs
        self.map_dimensions = (84, 64)
        self.history_length = history_length

        # screen conv layers
        self.screen_conv1 = nn.Conv2d(in_channels=1,
                                      out_channels=16,
                                      kernel_size=5,
                                      padding=0,
                                      stride=4)
        self.screen_conv2 = nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=3,
                                      padding=0,
                                      stride=1)
        # self.screen_conv3 = nn.Conv2d(in_channels=32
        #                                 out_channels=64,
        #                                 kernel_size=3,
        #                                 stride=1)

        # fully connected layers
        self.tmp_w = self._get_filter_dimension(84, 8, 0, 4)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, 4, 0, 2)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, 3, 0, 1)

        self.screen_fc1 = nn.Linear(64*self.tmp_w*self.tmp_w, 512)

        # action policy output
        self.out_fc1 = nn.Linear(512, self.num_outputs)

    def _get_filter_dimension(self, w, f, p, s):
        '''
        calculates filter dimension according to following formula:
        (filter - width + 2*padding) / stride + 1
        '''
        return int((w - f + 2*p) / s + 1)

    def forward(self, screen):
        screen = F.relu(self.screen_conv1(screen))
        screen = F.relu(self.screen_conv2(screen))
        screen = F.relu(self.screen_conv3(screen))
        screen = screen.view(-1, 64*self.tmp_w*self.tmp_w)
        screen = F.relu(self.screen_fc1(screen))

        # estimated action q values
        out_q_values = self.out_fc1(screen)

        return out_q_values

# if __name__ == '__main__':
#     AtariNet = AtariNet()
