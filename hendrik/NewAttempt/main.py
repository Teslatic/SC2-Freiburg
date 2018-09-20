#!/usr/bin/env python3

### pysc imports
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.env.environment import StepType

### torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


### normal python imports
from absl import app
import random
import numpy as np
from itertools import count
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'x_coord', 'y_coord', 'reward', 'next_state', 'step_type'))


# CONSTANTS
LEARNING_RATE = 0.01
GAMMA = 1.0
SCREEN_DIM = 84
MINIMAP_DIM = 64
GAMESTEPS = None # 0 = unlimited game time, None = map default
STEP_MULTIPLIER = 16 # 16 = 1s game time, None = map default
BATCH_SIZE = 256
TARGET_UPDATE =  10
VISUALIZE = False
NUM_EPISODES = 100000
MEMORY_SIZE = 10000
SMART_ACTIONS = [
          # actions.FUNCTIONS.select_army.id,
          actions.FUNCTIONS.Move_screen.id
        ]
SILENTMODE = False






def setup_agent():
  agent = BaseAgent(screen_dim = SCREEN_DIM, minimap_dim = MINIMAP_DIM,
          batch_size=BATCH_SIZE, TARGET_UPDATE=TARGET_UPDATE)
  # agent = scripted_agent.MoveToBeacon()
  # players = [ sc2_env.Agent(sc2_env.Race.terran),
  #       sc2_env.Bot(sc2_env.Race.random,
  #       sc2_env.Difficulty.very_easy)]
  players = [sc2_env.Agent(sc2_env.Race.terran)]
  agent_interface = features.AgentInterfaceFormat(
      feature_dimensions=features.Dimensions(screen=SCREEN_DIM,
                          minimap=MINIMAP_DIM),
      # rgb_dimensions = features.Dimensions(screen=SCREEN_DIM,
      #                     minimap=MINIMAP_DIM),
      # action_space = actions.ActionSpace.RGB,
      use_feature_units = True)

  return agent, players, agent_interface

def setup_env(agent, players, agent_interface):
  env = sc2_env.SC2Env(
      map_name="MoveToBeacon",
      players=players,
      agent_interface_format=agent_interface,
      step_mul=STEP_MULTIPLIER,
      game_steps_per_episode=GAMESTEPS,
      visualize=VISUALIZE)

  return env

def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))


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

class DQN(nn.Module):

  def __init__(self):
    super(DQN, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
    self.bn4 = nn.BatchNorm2d(128)
    self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2)

    self.head_actions = nn.Linear(5184, 1)
    self.head_x = nn.Linear(5184, 84)
    self.head_y = nn.Linear(5184, 64)


  def _get_filter_dimension(self,w,f,p,s):
    '''
    calculates filter dimension according to following formula:
    (filter - width + 2*padding) / stride + 1
    '''
    return int((w - f + 2*p) / s + 1)

  def forward(self, screen):
    # screen = F.relu(self.bn1(self.conv1(screen)))
    # screen = F.relu(self.bn2(self.conv2(screen)))
    # screen = F.relu(self.bn3(self.conv3(screen)))
    # screen = F.relu(self.bn4(self.conv4(screen)))

    screen = F.relu(self.conv1(screen))
    screen = F.relu(self.conv2(screen))
    screen = F.relu(self.conv3(screen))

    return self.head_actions(screen.view(screen.size(0), -1)), \
     self.head_x(screen.view(screen.size(0), -1)), \
     self.head_y(screen.view(screen.size(0), -1))



class BaseAgent(base_agent.BaseAgent):
  def __init__(self, screen_dim, minimap_dim, batch_size, TARGET_UPDATE):
    super(BaseAgent, self).__init__()

    self.net, self.target_net, self.optimizer = self._build_model()
    self.x_map_dim = 84
    self.y_map_dim = 64
    self.epsilon = 1.0
    self.eps_start = 1.0
    self.eps_end = 0.1
    self.eps_decay = 30000
    self.steps_done = 0
    self.choice = None
    print("Network: \n{}".format(self.net))
    print("Optimizer: \n{}".format(self.optimizer))
    print("Target Network: \n{}".format(self.target_net))

  def _build_model(self):
    net = DQN().to(device)
    target_net = DQN().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    return net, target_net, optimizer


  def initializing_timestep(self, obs, last_score):
      """
      Initializing variables and flags
      """
      self.reward += obs.reward
      self.step_type = obs.step_type
      self.available_actions = obs.observation.available_actions
      self.timesteps += 1
      self.steps += 1
      # self.actual_reward = obs.reward
      self.last_score = last_score
      self.feature_screen = obs.observation.feature_screen.player_relative
      self.beacon_center = np.mean(self._xy_locs(self.feature_screen == 3), axis=0).round()
      self.state = torch.tensor([self.feature_screen], dtype=torch.float, device=self.device, requires_grad = True)
      # self.history_tensor = self.hist.stack(self.state)  # stack state on history
      self.history_tensor = self.state
      self.state_history_tensor = self.state
      self.state_history_tensor = self.state.unsqueeze(1)

  def epsilon_greedy(self):
    """
    returns a string in order to determine if the next action choice is
    going to be random or according to an decaying epsilon greeedy policy
    """
    self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)
    self.steps_done += 1
    return np.random.choice(['random', 'greedy'], p=[self.epsilon, 1-self.epsilon])

  def step(self, obs):
    self.available_actions = obs.observation.available_actions
    self.feature_screen = obs.observation.feature_screen.player_relative
    self.beacon_center = np.mean(_xy_locs(self.feature_screen == 3), axis=0).round()
    self.state = torch.tensor([self.feature_screen], dtype=torch.float, device=device, requires_grad = True).unsqueeze(1)

    action, action_idx, x_coord, y_coord = self.choose_action(self.available_actions, self.state)

    return action, action_idx, x_coord, y_coord

  def choose_action(self, available_actions, state):
      self.choice = self.epsilon_greedy()
      if self.choice == 'random':
          action_idx = np.random.randint(len(SMART_ACTIONS))
          x_coord = np.random.randint(self.x_map_dim)
          y_coord = np.random.randint(self.y_map_dim)
          chosen_action = self.get_action(self.available_actions, SMART_ACTIONS[action_idx], x_coord, y_coord)
      else:
          with torch.no_grad():
            action_q_values, x_coord_q_values, y_coord_q_values = self.net(state)
          # action_idx = np.argmax(action_q_values)
          action_idx = 0
          best_action = SMART_ACTIONS[action_idx]
          x_coord = np.argmax(x_coord_q_values)
          y_coord = np.argmax(y_coord_q_values)
          chosen_action = self.get_action(self.available_actions, best_action, x_coord, y_coord)
      # square brackets arounc chosen_action needed for internal pysc2 state machine
      return [chosen_action], torch.tensor([action_idx], dtype=torch.long, device=device), \
       torch.tensor([x_coord], dtype=torch.long, device=device), \
        torch.tensor([y_coord], dtype=torch.long, device=device,)

  def get_action(self, available_actions, action, x, y):
    if self.can_do(available_actions, action):
        if action == actions.FUNCTIONS.Move_screen.id:
            return actions.FUNCTIONS.Move_screen("now", (x, y))
        # if action == actions.FUNCTIONS.select_army.id:
        #     return actions.FUNCTIONS.select_army("select")
        if action == actions.FUNCTIONS.no_op.id:
            return actions.FUNCTIONS.no_op()
    else:
        return actions.FUNCTIONS.no_op()

  def can_do(self, available_actions, action):
      return action in available_actions


def main(unused_argv):

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
    action_batch = torch.cat(batch.action).unsqueeze(1)
    x_batch = torch.cat(batch.x_coord).unsqueeze(1)
    y_batch = torch.cat(batch.y_coord).unsqueeze(1)
    reward_batch = torch.cat(batch.reward)

    q_action, q_x, q_y = agent.net(state_batch)
    q_action = q_action.gather(1, action_batch)
    q_x = q_x.gather(1, x_batch)
    q_y = q_y.gather(1, y_batch)

    q_action_next = torch.zeros(BATCH_SIZE, device=device)
    q_x_next = torch.zeros(BATCH_SIZE, device=device)
    q_y_next = torch.zeros(BATCH_SIZE, device=device)

    # q_action_next = reward_batch.detach()
    # q_x_next = reward_batch.detach()
    # q_y_next = reward_batch.detach()

    non_final_next_action_q ,non_final_next_x_q, non_final_next_y_q = agent.target_net(non_final_next_states)

    q_action_next[non_final_mask] = non_final_next_action_q.max(1)[0].detach()
    q_x_next[non_final_mask] = non_final_next_x_q.max(1)[0].detach()
    q_y_next[non_final_mask] = non_final_next_y_q.max(1)[0].detach()

    td_target_actions = (q_action_next * GAMMA) + reward_batch
    td_target_x = (q_x_next * GAMMA) + reward_batch
    td_target_y = (q_y_next * GAMMA) + reward_batch

    loss_actions = F.mse_loss(q_action, td_target_actions.unsqueeze(1))
    loss_x = F.mse_loss(q_x, td_target_x.unsqueeze(1))
    loss_y = F.mse_loss(q_y, td_target_y.unsqueeze(1))

    # print(loss_actions.item(), loss_x.item(), loss_y.item())

    loss = loss_actions + loss_x + loss_y


    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()


  total_reward = 0

  try:
    env = setup_env(agent,players,agent_interface)
    agent.setup(env.observation_spec(), env.action_spec())

    for e in range(1, NUM_EPISODES-1):
      observation = env.reset()
      actual_obs = observation[0]
      beacon = np.mean(_xy_locs(actual_obs.observation.feature_screen.player_relative == 3), axis=0).round()
      cnt = 0
      if SILENTMODE:
        print("---------------------------------------------------------------------")
        print("Episode: {}\t Total reward: {}".format(e,total_reward))
        print("Epsilon: {:.4f}".format(agent.epsilon))
        print("Beacon at {}".format(beacon))
        try:
          print(action)
        except:
          pass
        print("---------------------------------------------------------------------")

      while True:

        if actual_obs.first():
          action = [actions.FUNCTIONS.select_army("select")]
          action_idx = torch.tensor([0], device=device, dtype=torch.long)
          x_coord = torch.tensor([0], device=device, dtype=torch.long)
          y_coord = torch.tensor([0], device=device, dtype=torch.long)
        else:
          action, action_idx, x_coord, y_coord = agent.step(actual_obs)


        beacon = np.mean(_xy_locs(actual_obs.observation.feature_screen.player_relative == 3), axis=0).round()
        marine = np.mean(_xy_locs(actual_obs.observation.feature_screen.player_relative == 1), axis=0).round()

        b_x, b_y = beacon
        m_x, m_y = marine

        dx = m_x - b_x
        dy = m_y - b_y

        distance = np.sqrt(dx**2 + dy**2).round()
        scaling = lambda x : (x - 0)/(100 - 0)

        pseudo_reward = 1 - scaling(distance)

        reward = torch.tensor([pseudo_reward] , device=device, requires_grad=True, dtype=torch.float)
        total_reward += reward


        if not SILENTMODE:
          print("---------------------------------------------------------------------")
          print("Episode: {}\t Total reward: {}".format(e,total_reward))
          print("Epsilon: {:.4f}".format(agent.epsilon))
          print("Beacon at {}".format(beacon))
          print(action)
          print("---------------------------------------------------------------------")


        next_obs = env.step(action)

        # reward = torch.tensor([actual_obs.reward], device=device, requires_grad=True, dtype=torch.float)
        # total_reward += actual_obs.reward

        if next_obs[0].last():
          next_state = None
        else:
          next_state = torch.tensor([next_obs[0].observation.feature_screen.player_relative],\
            dtype=torch.float, device=device, requires_grad = True).unsqueeze(1)

        state = torch.tensor([actual_obs.observation.feature_screen.player_relative],\
         dtype=torch.float, device=device, requires_grad = True).unsqueeze(1)

        memory.push(state, action_idx, x_coord, y_coord, reward , next_state, next_obs[0].step_type.value)

        actual_obs = next_obs[0]

        optimize_model()

        if e % TARGET_UPDATE == 0:
          agent.target_net.load_state_dict(agent.net.state_dict())

        if next_obs[0].last():
          break

  except KeyboardInterrupt:
    pass








if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("Device: {}".format(device))
  agent, players, agent_interface = setup_agent()
  memory = ReplayBuffer(MEMORY_SIZE)
  app.run(main)
