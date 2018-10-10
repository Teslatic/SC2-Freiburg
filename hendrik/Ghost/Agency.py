__author__ = "Hendrik Vloet"
__copyright__ = "Copyright (C) 2018 Hendrik Vloet"
__license__ = "Public Domain"
__version__ = "1.0"

# ______________________________________________________________________________

# pysc imports
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.env.environment import StepType

# normal python imports
from absl import app
from absl import flags
from absl import logging
import random
import numpy as np
import math
from itertools import count
from collections import namedtuple

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

## @package Agency
#  Documentation for Agency.
#
#  More details.

## Documentation for ReplayBuffer
#
#  More details.
class ReplayBuffer(object):
    ''' Experience Replay Buffer Class '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
            ('state', 'action', 'reward', 'next_state',
             'step_type'))

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, key):
        return self.memory[key]

    def push(self, *args):
        ''' store transition in replay buffer '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        ''' return random sample transition from ER '''
        return random.sample(self.memory, batch_size)

## Documentation for BaseAgent
#
#  More details.
class BaseAgent(base_agent.BaseAgent):
    """ BaseAgent class
    This is the central agent which controls all program flow comprising:
      - parsing flags to hyperparameter member variables
      - building the net, target net and optimizer
      - setting up the pysc2 environment in order to use it
      - playing
      - training
      - optimizing
    """
    def __init__(self,architecture, FLAGS, device="cpu"):
        super(BaseAgent, self).__init__()
        # parsing arguments to hyperparameter
        self._lr = FLAGS.learning_rate
        self._gamma = FLAGS.learning_rate
        self._batch_size = FLAGS.batch_size
        self._target_update = FLAGS.target_update
        self._epochs = FLAGS.epochs
        self._memory_size = FLAGS.memory_size
        self._visualize = FLAGS.visualize
        self._device = device
        self._map_name = FLAGS.map_name
        self._step_multiplier = FLAGS.step_multiplier

        self._eps_decay = FLAGS.epsilon
        self.epsilon = 1.0
        self._EPS_START = 1.0
        self._EPS_END = 0.1
        self._steps_done = 0
        self._choice = None

        self.total_reward = 0
        self._SMART_ACTIONS = [
            actions.FUNCTIONS.select_army.id,
            actions.FUNCTIONS.Move_screen.id ]

        self.memory = ReplayBuffer(self._memory_size)

        self._net, self._target_net, self._optimizer = \
            self._build_model(architecture)

        self._env = self._build_env()

        self._xy_pairs = self._discretize_xy_grid(FLAGS.xy_grid)

        print("Network: \n{}".format(self._net))
        print("Optimizer: \n{}".format(self._optimizer))
        print("Target Network: \n{}".format(self._target_net))

    def _build_model(self, architecture):
        net = architecture.to(self._device)
        target_net = architecture.to(self._device)
        optimizer = optim.Adam(net.parameters(), lr=self._lr)

        return net, target_net, optimizer

    def _build_env(self):
        players = [sc2_env.Agent(sc2_env.Race.terran)]
        interface = features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84,
                minimap=84), use_feature_units = True)
        env = sc2_env.SC2Env(
            map_name = self._map_name,
            players = players,
            agent_interface_format = interface,
            step_mul = self._step_multiplier,
            game_steps_per_episode = 0,
            visualize = self._visualize)

        return env

    def _discretize_xy_grid(self, factor):
        """ "discretizing" action coordinates in order to keep action space small """
        x_space = np.linspace(0, 83, factor, dtype = int)
        y_space = np.linspace(0, 63, factor, dtype = int)
        xy_space = np.transpose([np.tile(x_space, len(y_space)),
                                   np.repeat(y_space, len(x_space))])

        return xy_space

    def _xy_locs(self, mask):
        """Mask should be a set of bools from comparison with a feature layer."""
        y, x = mask.nonzero()

        return list(zip(x, y))

    def decide(self):
        """ returns a string in order to determine if the next action choice is
        going to be random or according to an decaying epsilon greeedy policy
        """
        self.epsilon = self._EPS_END + (self._EPS_START - self._EPS_END) \
            * np.exp(-1. * self._steps_done / self._eps_decay)
        self._steps_done += 1
        self.choice = np.random.choice(['random', 'greedy'],
                                       p=[self.epsilon, 1-self.epsilon])

    def choose_action(self):
        self.decide()
        if self.choice == 'random':
            self.action_idx = torch.tensor([random.randint(0, len(self._xy_pairs) - 1)],
                                           dtype=torch.long, device=self._device)
            self.action_xy = self._xy_pairs[self.action_idx]
        else:
            with torch.no_grad():
                self.action_q_values = self._net(self.state)
                self.action_idx = torch.tensor([np.argmax(self.action_q_values)],
                                           dtype=torch.long, device=self._device)
                self.action_xy = self._xy_pairs[self.action_idx]
        self.x_coord = self.action_xy[0]
        self.y_coord = self.action_xy[1]


        if self.actual_obs.first()==True:
            self.pysc_action = self._SMART_ACTIONS[0]
        else:
            self.pysc_action = self._SMART_ACTIONS[1]

        self.extract_action()


    def extract_action(self):
        if self.pysc_action in self.actual_obs.observation.available_actions:
            if self.pysc_action == 331:
                self.action = [actions.FUNCTIONS.Move_screen("now",
                                        (self.x_coord, self.y_coord))]
            if self.pysc_action == 7:
                self.action = [actions.FUNCTIONS.select_army("select")]
            if self.pysc_action == 0:
                self.action = [actions.FUNCTIONS.no_op()]
        else:
            self.action = [actions.FUNCTIONS.no_op()]

    def step(self):
        self.choose_action()

    def calc_pseudo_reward(self):
            self.beacon = np.mean(self._xy_locs(
                self.actual_obs.observation.feature_screen.player_relative == 3),
                             axis=0).round()
            b_x, b_y = self.beacon
            if self.actual_obs.first()==False:
                try:
                    marine = self._xy_locs(self.actual_obs.observation.feature_screen.selected == 1)
                    marine = np.mean(marine, axis = 0).round()
                    self.m_x = marine[0]
                    self.m_y = marine[1]

                    dx = self.m_x - b_x
                    dy = self.m_y - b_y

                    distance = np.sqrt(dx**2 + dy**2).round()
                    scaling = lambda x : (x - 0)/(100 - 0)

                    self.pseudo_reward = (0.1 * (1 - scaling(distance))).round(2)

                    if self.next_obs[0].reward==1:
                        self.reward = torch.tensor([self.pseudo_reward + 10] , device=self._device,
                                       requires_grad=True, dtype=torch.float)
                    else:
                        self.reward = torch.tensor([self.pseudo_reward] , device=self._device,
                                                   requires_grad=True, dtype=torch.float)
                except:
                    pass
            else:
                self.m_x = -1
                self.m_y = -1
                self.reward = torch.tensor([0] , device=self._device,
                                       requires_grad=True, dtype=torch.float)

            self.total_reward += self.actual_obs.reward

    def optimize(self):
        if len(self.memory) < self._batch_size:
            return
        transitions = self.memory.sample(self._batch_size)

        batch = self.memory.Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                batch.next_state)), device=self._device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)

        q_action = self._net(state_batch).gather(1, action_batch)

        q_action_next = torch.zeros(self._batch_size, device=self._device)
        non_final_next_action_q  = self._target_net(non_final_next_states)

        q_action_next[non_final_mask] = non_final_next_action_q.max(1)[0].detach()

        td_target_actions = (q_action_next * self._gamma) + reward_batch

        loss = F.mse_loss(q_action, td_target_actions.unsqueeze(1))

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss



    def print_status(self):

        print("Episode: {}\t Total pysc2 reward: {}\t Pseudo Reward: {}".format(
            self.e,self.total_reward,self.reward))
        print("Epsilon: {:.4f}".format(self.epsilon))
        print("Beacon at {}".format(self.beacon))
        print("Marine at {}".format((self.m_x, self.m_y)))
        print("Loss: {}".format(self.loss))
        print(self.action)
        print(100 * "=")

    def play(self):
        for self.e in range(1, self._epochs):
            observation = self._env.reset()
            self.actual_obs = observation[0]
            while (True):
                self.state = torch.tensor([self.actual_obs.observation.feature_screen.player_relative],
                                      dtype=torch.float, device=self._device, requires_grad = True).unsqueeze(1)

                # agent deterines action to take
                self.step()
                # actual pysc2 environment executes chosen action
                self.next_obs = self._env.step(self.action)

                self.calc_pseudo_reward() 

                if self.next_obs[0].last():
                    self.next_state = None
                else:
                    self.next_state = torch.tensor([self.next_obs[0].observation.feature_screen.player_relative],\
                                          dtype=torch.float, device=self._device,
                                               requires_grad = True).unsqueeze(1)

                self.memory.push(self.state, self.action_idx, self.reward , self.next_state,
                        self.next_obs[0].step_type.value)

                self.actual_obs = self.next_obs[0]

                self.loss = self.optimize()

                self.print_status()

                if self.e % self._target_update == 0:
                    self._target_net.load_state_dict(self._net.state_dict())


                if self.next_obs[0].last()==True:
                    break
