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


class ExtendedDQN(nn.Module):

    def __init__(self, history_length, dim_actions):
        super(ExtendedDQN, self).__init__()
        self.num_actions = dim_actions
        self.map_dimensions = (84, 64)
        self.history_length = history_length

        # CNN Layer properties
        KERNEL_1 = 13
        STRIDE_1 = 1
        PADDING_1 = 0

        KERNEL_2 = 5
        STRIDE_2 = 2
        PADDING_2 = 0

        KERNEL_3 = 3
        STRIDE_3 = 1
        PADDING_3 = 0

        POOL_PAD = 0
        POOL_STR = 2
        POOL_KRL = 2

        # screen conv layers
        self.screen_conv1 = nn.Conv2d(in_channels=3,
                                      out_channels=32,
                                      kernel_size=KERNEL_1,
                                      padding=PADDING_1,
                                      stride=STRIDE_1)

        self.bn1 = nn.BatchNorm2d(32)

        self.screen_conv2 = nn.Conv2d(in_channels=32,
                                      out_channels=64,
                                      kernel_size=KERNEL_2,
                                      padding=PADDING_2,
                                      stride=STRIDE_2)

        self.bn2 = nn.BatchNorm2d(64)

        self.screen_conv3 = nn.Conv2d(in_channels=64,
                                        out_channels=128,
                                        kernel_size=KERNEL_3,
                                        padding=PADDING_3,
                                        stride=STRIDE_3)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(p=0.5)

        # fully connected layers
        self.tmp_w = self._get_filter_dimension(200, KERNEL_1, PADDING_1, STRIDE_1)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, POOL_KRL,POOL_PAD,POOL_STR)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, KERNEL_2, PADDING_2, STRIDE_2)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, POOL_KRL,POOL_PAD,POOL_STR)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, KERNEL_3, PADDING_3, STRIDE_3)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, POOL_KRL,POOL_PAD,POOL_STR)

        self.tmp_h = self._get_filter_dimension(400 , KERNEL_1, PADDING_1, STRIDE_1)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, POOL_KRL,   POOL_PAD, POOL_STR)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, KERNEL_2, PADDING_2, STRIDE_2)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, POOL_KRL,   POOL_PAD, POOL_STR)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, KERNEL_3, PADDING_3, STRIDE_3)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, POOL_KRL,   POOL_PAD, POOL_STR)


        self.screen_fc1 = nn.Linear(128*self.tmp_w*self.tmp_h, 256)
        self.screen_fc2 = nn.Linear(256, 512)
        self.screen_fc3 = nn.Linear(512, self.num_actions)

        self.softmax = nn.Softmax(dim=1)

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

    def forward(self, screen):
        screen = self.pool(F.relu(self.bn1(self.screen_conv1(screen))))
        screen = self.pool(F.relu(self.bn2(self.screen_conv2(screen))))
        screen = self.pool(F.relu(self.bn3(self.screen_conv3(screen))))
        # screen = self.screen_fc1(screen.view(screen.size(0),-1))
        screen = screen.view(-1, self.num_flat_features(screen))
        screen = self.dropout(F.relu(self.screen_fc1(screen)))
        screen = self.dropout(F.relu(self.screen_fc2(screen)))
        action_q_values =  F.relu(self.screen_fc3(screen))
        # print(action_q_values)
        # action_q_values = F.relu(self.head(screen))
        output = self.softmax(action_q_values)
        return output

class DQN(nn.Module):

    def __init__(self, history_length, dim_actions):
        super(DQN, self).__init__()
        self.num_actions = dim_actions
        self.map_dimensions = (84, 64)
        self.history_length = history_length

        # FC Layer properties
        KERNEL_1 = 3
        STRIDE_1 = 2
        PADDING_1 = 0

        KERNEL_2 = 3
        STRIDE_2 = 2
        PADDING_2 = 0

        KERNEL_3 = 3
        STRIDE_3 = 2
        PADDING_3 = 0

        POOL_PAD = 0
        POOL_STR = 2
        POOL_KRL = 2

        # screen conv layers
        self.screen_conv1 = nn.Conv2d(in_channels=3,
                                      out_channels=16,
                                      kernel_size=KERNEL_1,
                                      padding=PADDING_1,
                                      stride=STRIDE_1)

        self.bn1 = nn.BatchNorm2d(16)

        self.screen_conv2 = nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=KERNEL_2,
                                      padding=PADDING_2,
                                      stride=STRIDE_2)

        self.bn2 = nn.BatchNorm2d(32)

        self.screen_conv3 = nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=KERNEL_3,
                                        padding=PADDING_3,
                                        stride=STRIDE_3)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(p=0.5)

        # fully connected layers
        self.tmp_w = self._get_filter_dimension(200, KERNEL_1, PADDING_1, STRIDE_1)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, POOL_KRL,POOL_PAD,POOL_STR)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, KERNEL_2, PADDING_2, STRIDE_2)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, POOL_KRL,POOL_PAD,POOL_STR)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, KERNEL_3, PADDING_3, STRIDE_3)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, POOL_KRL,POOL_PAD,POOL_STR)

        self.tmp_h = self._get_filter_dimension(400 , KERNEL_1, PADDING_1, STRIDE_1)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, POOL_KRL,   POOL_PAD, POOL_STR)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, KERNEL_2, PADDING_2, STRIDE_2)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, POOL_KRL,   POOL_PAD, POOL_STR)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, KERNEL_3, PADDING_3, STRIDE_3)
        self.tmp_h = self._get_filter_dimension(self.tmp_h, POOL_KRL,   POOL_PAD, POOL_STR)

        self.screen_fc1 = nn.Linear(64*self.tmp_w*self.tmp_h, 128)
        self.screen_fc2 = nn.Linear(128, 256)
        self.screen_fc3 = nn.Linear(256, self.num_actions)

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

    def forward(self, screen):
        screen = self.pool(F.relu(self.bn1(self.screen_conv1(screen))))
        screen = self.pool(F.relu(self.bn2(self.screen_conv2(screen))))
        screen = self.pool(F.relu(self.bn3(self.screen_conv3(screen))))
        # screen = self.screen_fc1(screen.view(screen.size(0),-1))
        # print(screen.shape)
        # exit()
        screen = screen.view(-1, self.num_flat_features(screen))
        screen = self.dropout(F.relu(self.screen_fc1(screen)))
        screen = self.dropout(F.relu(self.screen_fc2(screen)))
        action_q_values = F.relu(self.screen_fc3(screen))
        # print(action_q_values)
        # action_q_values = F.relu(self.head(screen))
        return action_q_values
