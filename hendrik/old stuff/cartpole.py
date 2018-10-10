#!/usr/bin/env python3


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class TutorialDQN(nn.Module):

    def __init__(self):
        super(TutorialDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class SingleDQN(nn.Module):

  def __init__(self):
    super(SingleDQN, self).__init__()

    # screen conv layers
    self.screen_conv1 = nn.Conv2d(3, 16, kernel_size=8, padding=0, stride =4)
    self.screen_conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=0, stride=2)
    self.screen_conv3 = nn.Conv2d(32,64, kernel_size=3, stride=1)

    # fully connected layers
    # self.tmp_w = self._get_filter_dimension(84, 8, 0, 4)
    # self.tmp_w = self._get_filter_dimension(self.tmp_w, 4, 0,2)
    # self.tmp_w = self._get_filter_dimension(self.tmp_w,3, 0,1)
    #
    # self.screen_fc1 = nn.Linear(64*self.tmp_w*self.tmp_w,512)

    # action policy output
    self.out_fc1 = nn.Linear(384,2)

  def _get_filter_dimension(self,w,f,p,s):
    '''
    calculates filter dimension according to following formula:
    (filter - width + 2*padding) / stride + 1
    '''
    return int((w - f + 2*p) / s + 1)

  def forward(self, screen):
    screen = F.relu(self.screen_conv1(screen))
    screen = F.relu(self.screen_conv2(screen))
    screen = F.relu(self.screen_conv3(screen))
    # screen = screen.view(-1, 64*self.tmp_w*self.tmp_w)
    # screen = F.relu(self.screen_fc1(screen))

    # estimated action q values
    # out_q_values = self.out_fc1(screen)


    return self.out_fc1(screen.view(screen.size(0), -1))




def select_action(state):
    global steps_done
    global eps_threshold
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())






def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)



resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def optimize_model():
  if len(memory) < BATCH_SIZE:
      return
  transitions = memory.sample(BATCH_SIZE)
  batch = Transition(*zip(*transitions))

  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                batch.next_state)), device=device, dtype=torch.uint8)
  non_final_next_states = torch.cat([s for s in batch.next_state
                if s is not None])

  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)


  q_action = net(state_batch).gather(1, action_batch)

  q_action_next = torch.zeros(BATCH_SIZE, device=device)

  non_final_next_action_q = target_net(non_final_next_states)

  q_action_next[non_final_mask] = non_final_next_action_q.max(1)[0].detach()

  td_target_actions = (q_action_next * GAMMA) + reward_batch

  loss_actions = F.mse_loss(q_action, td_target_actions.unsqueeze(1))


  loss = loss_actions


  optimizer.zero_grad()
  loss.backward()
  optimizer.step()




# This is based on the code from gym.
screen_width = 600
steps_done = 0
memory = ReplayMemory(10000)


NUM_EPISODES = 500
BATCH_SIZE = 120
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 10

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

net = SingleDQN().to(device)
target_net = SingleDQN().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0002)

target_net.load_state_dict(net.state_dict())
target_net.eval()

env = gym.make('CartPole-v0').unwrapped
total_reward = 0
for e in range(1, NUM_EPISODES-1):
  env.reset()
  last_screen = get_screen()
  current_screen = get_screen()
  state = current_screen - last_screen
  cnt = 0
  reward_per_epoch = 0
  while True:
    try:
      print("----------------------------------------------------------------------------------------")
      print("Epoch {}\tStep {}\tReward per Epoch {}\tTotal Reward {}\t Epsilon {:.4f}".format(e, steps_done ,reward_per_epoch, total_reward, eps_threshold))
      print("----------------------------------------------------------------------------------------")
    except:
      pass
    action = select_action(state)
    _, reward, done, _ = env.step(action.item())
    total_reward += reward
    reward_per_epoch += reward
    reward = torch.tensor([reward], device=device)
    # Observe new state
    last_screen = current_screen
    current_screen = get_screen()

    if not done:
        next_state = current_screen - last_screen
    else:
        next_state = None

    memory.push(state, action, next_state, reward)

    state = next_state
    optimize_model()
    cnt += 1
    if done:
      episode_durations.append(cnt + 1)
      plot_durations()
      break

  if e % TARGET_UPDATE == 0:
      target_net.load_state_dict(net.state_dict())
