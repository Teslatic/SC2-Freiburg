#!/usr/bin/env python3

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# assign operations to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
AtariNet accordingly to the SC2-paper
Inputs:
    - non-spatial features
    - screen
    - minimap

The exact architecture, shown schematically in Fig. 1, is as follows. The input to
the neural network consists of an 84
3
84
3
4 image produced by the preprocess-
ing map
w
. The first hidden layer convolves 32 filters of 8
3
8 with stride 4 with the
input image and applies a rectifier nonlinearity
31,32
. The second hidden layer con-
volves 64 filters of 4
3
4 with stride 2, again followed by a rectifier nonlinearity.
This is followedby a thirdconvolutional layer thatconvolves 64filtersof3
3
3 with
stride 1 followed by a rectifier. The final hidden layer is fully-connected and con-
sists of 512 rectifier units. The output layer is a fully-connected linear layer with a
single output for each valid action. The number of valid actions varied between 4
and 18 on the games we considered.
'''


class DQN(nn.Module):

    def __init__(self, history_length):
        super(DQN, self).__init__()
        self.num_actions = 4
        self.map_dimensions = (84, 64)
        self.history_length = history_length

        # screen conv layers
        self.screen_conv1 = nn.Conv2d(1, 16, kernel_size=8, padding=0, stride=4)
        self.screen_conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=0, stride=2)
        self.screen_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # minimap conv layers
        self.minimap_conv1 = nn.Conv2d(3, 16, kernel_size=8, padding=0, stride=4)
        self.minimap_conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=0, stride=2)
        self.minimap_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # fully connected layers
        self.tmp_w = self._get_filter_dimension(84, 8, 0, 4)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, 4, 0, 2)
        self.tmp_w = self._get_filter_dimension(self.tmp_w, 3, 0, 1)

        self.screen_fc1 = nn.Linear(64*self.tmp_w*self.tmp_w, 512)

        # simple policy output
        self.action_fc1 = nn.Linear(512, self.num_actions)

        # # action policy output
        # self.action_fc1 = nn.Linear(512, self.num_actions)
        #
        # # x coordinate output
        # self.x_coord_fc1 = nn.Linear(512, self.map_dimensions[0])
        #
        # # y coordinate output
        # self.y_coord_fc1 = nn.Linear(512, self.map_dimensions[1])

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
        action_q_values = self.action_fc1(screen)

        # estimated x coordinates q values
        x_coord_q_values = self.x_coord_fc1(screen)

        # estimated y coordinates q values
        y_coord_q_values = self.y_coord_fc1(screen)

        return action_q_values, x_coord_q_values, y_coord_q_values

    def forward(self, screen):
        screen = F.relu(self.screen_conv1(screen))
        screen = F.relu(self.screen_conv2(screen))
        screen = F.relu(self.screen_conv3(screen))
        screen = screen.view(-1, 64*self.tmp_w*self.tmp_w)
        screen = F.relu(self.screen_fc1(screen))

        # estimated action q values
        action_q_values = self.action_fc1(screen)
        #
        # # estimated x coordinates q values
        # x_coord_q_values = self.x_coord_fc1(screen)

        # # estimated y coordinates q values
        # y_coord_q_values = self.y_coord_fc1(screen)

        return action_q_values

class SingleDQN(nn.Module):

    def __init__(self, history_length, num_outputs):
        super(SingleDQN, self).__init__()
        self.num_outputs = num_outputs
        self.map_dimensions = (84, 64)
        self.history_length = history_length

        # screen conv layers
        self.screen_conv1 = nn.Conv2d(self.history_length, 16, kernel_size=8, padding=0, stride=4)
        self.screen_conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=0, stride=2)
        self.screen_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

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


if __name__ == '__main__':
    AtariNet = AtariNet()
