
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
import os
import pandas as pd
from pathlib import Path

# # torch imports
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

# custom imports
from assets.RL.DQN_module import DQN_module
from assets.smart_actions import SMART_ACTIONS_SIMPLE_NAVIGATION as SMART_ACTIONS
from assets.helperFunctions.flagHandling import set_flag_every, set_flag_from
from assets.helperFunctions.timestamps import print_timestamp as print_ts
from assets.helperFunctions.FileManager import *

## Documentation for BaseAgent
#
#  This is the central agent which controls all program flow comprising:
#  - parsing flags to hyperparameter member variables
#  - building the net, target net and optimizer
#  - setting up the pysc2 environment in order to use it
#  - playing
#  - training
#  - optimizing
class GridAgent(base_agent.BaseAgent):
    ## Constructor
    def __init__(self, agent_file):
        super(GridAgent, self).__init__()
        self.unzip_hyperparameter_file(agent_file)

        self.steps_done = 0  # total_timesteps
        self.timesteps = 0  # timesteps in the current episode
        self.choice = None  # Choice of epsilon greedy
        self.loss = 0  # Action loss
        self.reward = 0
        self.episode_reward_env = 0
        self.episode_reward_shaped = 0
        self._xy_pairs = self.discretize_xy_grid()
        self.dim_actions = len(self._xy_pairs)
        self.DQN = DQN_module(self.batch_size,
                              self.gamma,
                              self.history_length,
                              self.size_replaybuffer,
                              self.optim_learning_rate,
                              self.dim_actions
        )
        self.device = self.DQN.device
        print_ts("Agent has been initalized")


    def unzip_hyperparameter_file(self, agent_file):
        """
        Unzipping the hyperparameter files and writing them into member variables.
        """
        self.action_dim = len(SMART_ACTIONS)
        self.gamma = agent_file['GAMMA']
        self.optim_learning_rate = agent_file['OPTIM_LR']
        self.batch_size = agent_file['BATCH_SIZE']
        self.target_update_period = agent_file['TARGET_UPDATE_PERIOD']
        self.history_length = agent_file['HIST_LENGTH']
        self.size_replaybuffer = agent_file['REPLAY_SIZE']
        self.device = agent_file['DEVICE']
        # self.silentmode = agent_file['SILENTMODE']
        # self.logging = agent_file['LOGGING']
        self.supervised_episodes = agent_file['SUPERVISED_EPISODES']
        self.patience = agent_file['PATIENCE']


        epsilon_file = agent_file['EPSILON_FILE']
        self.epsilon = epsilon_file['EPSILON']
        self.eps_start = epsilon_file['EPS_START']
        self.eps_end = epsilon_file['EPS_END']
        self.eps_decay = epsilon_file['EPS_DECAY']

        self.exp_path = create_experiment_at_main(agent_file['EXP_PATH'])

        self.grid_factor = agent_file['GRID_FACTOR']

    # ##########################################################################
    # Action Selection
    # ##########################################################################

    def prepare_timestep(self, obs, reward, done, info):
        """
        timesteps:
        """
        # from PYSC2 base class
        self.steps += 1
        self.reward += reward

        # Current episode
        self.timesteps += 1
        self.episode_reward_env += reward
        # TODO(vloeth): extend observation to full pysc2 observation 
        # self.available_actions = obs.observation.available_actions

        # Calculate additional information for reward shaping
        self.state = obs[0]
        # self.beacon_center, self.marine_center, self.distance = self.calculate_distance(self.feature_screen, self.feature_screen2)
        self.last = obs[3]
        self.first = obs[2]
        self.distance = obs[4]
        self.marine_center  = obs[5]
        self.beacon_center = obs[6]


    def policy(self, obs, reward, done, info):
        """
        Choosing an action
        """
        # Set all variables at the start of a new timestep
        self.prepare_timestep(obs, reward, done, info)

        # Action seletion according to active policy
        # action = [actions.FUNCTIONS.select_army("select")]

        if self.first:  # Select Army in first step
            return [actions.FUNCTIONS.select_army("select")]

        if self.last:  # End episode in last step
            print_ts("Last step: epsilon is at {}, Total score is at {}".format(self.epsilon, self.reward))
            self._save_model()
            self.update_target_network()
            self.action = 'reset'

        # Action selection for regular step
        if not self.first and not self.last:
            # For the first n episodes learn on forced actions.
            if self.episodes < self.supervised_episodes:
                self.action, self.action_idx = self.supervised_action()
            else:
                # Choose an action according to the policy
                self.action, self.action_idx = self.choose_action()
        return self.action_idx



    def supervised_action(self):

        # self.beacon = np.mean(self._xy_locs(
                # self.actual_obs.observation.feature_screen.player_relative == 3),
                             # axis=0).round()

            self.x_coord = self.beacon_center[0]
            self.y_coord = self.beacon_center[1]

            distances = []
            for xy_pair in self._xy_pairs:
                dx = np.abs(xy_pair[0] - self.beacon_center[0])
                dy = np.abs(xy_pair[1] - self.beacon_center[1])
                distances.append(np.sqrt(dx**2 + dy**2).round())

            closest_pair = np.argmin(distances)
            # print("Closes pair: {}".format(closest_pair))
            self.action_idx = closest_pair
            return self.action_idx, self.action_idx


    def choose_action(self, agent_mode='learn'):
        """
        chooses an action according to the current policy
        returns the chosen action id with x,y coordinates and the index of the action
        with respect to the SMART_ACTIONS constant. The additional index is used for
        the catergorical labeling of the actions as an "intermediate" solution for
        a restricted action space.
        """
        self.choice = self.epsilon_greedy()

        if self.choice == 'random' and agent_mode == 'learn':
            action_idx = np.random.randint(self.action_dim)
            # chosen_action = self.translate_to_PYSC2_action(SMART_ACTIONS[action_idx])
        else:
            action_q_values = self.DQN.predict_q_values(self.state)

            # Beste Action bestimmen
            best_action_numpy = action_q_values.detach().cpu().numpy()
            action_idx = np.argmax(best_action_numpy)
            best_action = action_idx

            # chosen_action = self.translate_to_PYSC2_action(best_action)
        # square brackets around chosen_action needed for internal pysc2 state machine
        return [action_idx], action_idx

    def epsilon_greedy(self):
        """
        returns a string in order to determine if the next action choice is
        going to be random or according to an decaying epsilon greeedy policy
        """
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        return np.random.choice(['random', 'greedy'], p=[self.epsilon, 1-self.epsilon])

    def store_transition(self, next_obs, reward):
        """
        Save the actual information in the history.
        As soon as there has been enough data, the experience is sampled from the replay buffer.
        """
        # don't store transition if first or last step
        if self.first or self.last:
            return

        self.reward = reward
        # self.episode_reward_shaped += self.reward_shaped
        self.next_state = next_obs[0]

        # Maybe using uint8 for storing into ReplayBuffer
        # self.state_uint8 = np.uint8(self.state)
        # self.next_state_uint8 = np.uint8(self.next_state)
        # self.action_idx_uint8 = np.uint8(self.action_idx)

        # save transition tuple to the memory buffer
        # self.DQN.memory.push([np.uint8(self.state)], [self.action_idx], self.reward_shaped, [np.uint8(self.next_state)])
        self.DQN.memory.push([self.state], [self.action_idx], self.reward, [self.next_state])


    ## "discretizes" the x,y coordinate system into smaller parts in order to keep the
    #  action space smaller
    #
    #  @param[in] factor: splits the total original x,y grid into factor^2 pairs
    #  @param[out] xy_space: array containing the discretized x,y coordinate pairs
    def discretize_xy_grid(self):
        """ "discretizing" action coordinates in order to keep action space small """
        x_space = np.linspace(0, 83, self.grid_factor, dtype = int)
        y_space = np.linspace(0, 63, self.grid_factor, dtype = int)
        xy_space = np.transpose([np.tile(x_space, len(y_space)),
                                   np.repeat(y_space, len(x_space))])

        return xy_space


    # ## chooses the next action
    # #
    # #  depends of the outcom of decide():
    # #  - random: from the list of _xy_pairs, one pair is chosen and applied to the
    # #  move screeen action
    # #  - greedy: current state is feedforwarded through the network and a x,y pair
    # #  is picked according to the max q value
    # def choose_action(self):
    #     if self.e % self._imitation_phase_length == 0:
    #         self._imitation_phase = False

    #     if self._imitation_phase==False:
    #         self.decide()
    #         if self.choice == 'random':
    #             ## the index of the x,y pair that is chosen
    #             self.action_idx = torch.tensor([random.randint(0, len(self._xy_pairs) - 1)],
    #                                        dtype=torch.long, device=self._device)
    #             ## the actual x,y pair which will be used in the next action
    #             self.action_xy = self._xy_pairs[self.action_idx]
    #         else:
    #             with torch.no_grad():
    #                 self.action_q_values = self._net(self.state)
    #                 self.action_idx = torch.tensor([np.argmax(self.action_q_values)],
    #                                        dtype=torch.long, device=self._device)
    #                 self.action_xy = self._xy_pairs[self.action_idx]
    #         self.x_coord = self.action_xy[0]
    #         self.y_coord = self.action_xy[1]
    #         self.list_x.append(self.x_coord)
    #         self.list_y.append(self.y_coord)
    #     else:
    #         self.beacon = np.mean(self._xy_locs(
    #             self.actual_obs.observation.feature_screen.player_relative == 3),
    #                          axis=0).round()

    #         self.x_coord = self.beacon[0]
    #         self.y_coord = self.beacon[1]

    #         # print("Beacon target: {},{}".format(self.x_coord, self.y_coord))
    #         # print("xy space: {}".format(self._xy_pairs))

    #         distances = []
    #         for xy_pair in self._xy_pairs:
    #             dx = np.abs(xy_pair[0] - self.beacon[0])
    #             dy = np.abs(xy_pair[1] - self.beacon[1])
    #             distances.append(np.sqrt(dx**2 + dy**2).round())

    #         closest_pair = np.argmin(distances)
    #         # print("Closes pair: {}".format(closest_pair))
    #         self.action_idx = torch.tensor([closest_pair],
    #                                        dtype=torch.long, device=self._device)


    #     # first action is always to select the army
    #     if self.actual_obs.first()==True:
    #         ## the action id of the pysc2 action, necessary to extract the real action
    #         self.pysc_action = self._SMART_ACTIONS[0]
    #     else:
    #         self.pysc_action = self._SMART_ACTIONS[1]

    #     self.extract_action()


    ## extracts the action from the list of available actions in the actual game
    #
    #  the game needs specific encoded actions in order to work. These functions
    #  are too complicated for a network to handle, so the network would (in the later
    #  versions of the agent) pick numbers which are in turn ENUMS.
    #  According to the picked number, the unique action is applied
    #
    #  errorous action choices always result in NO_OP
    def extract_action(self):
        if self.pysc_action in self.actual_obs.observation.available_actions:
            if self.pysc_action == 331:
                ## the actual action which is used by the pysc2 enginge
                self.action = [actions.FUNCTIONS.Move_screen("now",
                                        (self.x_coord, self.y_coord))]
            if self.pysc_action == 7:
                self.action = [actions.FUNCTIONS.select_army("select")]
            if self.pysc_action == 0:
                self.action = [actions.FUNCTIONS.no_op()]
        else:
            self.action = [actions.FUNCTIONS.no_op()]




    # ##########################################################################
    # DQN module wrappers
    # ##########################################################################

    def get_memory_length(self):
        """
        Returns the length of the ReplayBuffer
        """
        return len(self.DQN.memory)

    def optimize(self):
        """
        Optimizes the DQN_module on a minibatch
        """
        if self.get_memory_length() >= self.batch_size * self.patience:
            self.DQN.optimize()

    def log(self):
        pass
        buffer_size = 10 # This makes it so changes appear without buffering
        with open('output.log', 'w', buffer_size) as f:
                f.write('{}\n'.format(self.feature_screen))

    def _save_model(self, emergency=False):
        if emergency:
            save_path = self.exp_path + "/model/emergency_model.pt"
        else:
            save_path = self.exp_path + "/model/model.pt"

        # Path((self.exp_path + "/model")).mkdir(parents=True, exist_ok=True)
        self.DQN.save(save_path)




 # ##########################################################################
    # Ending Episode
    # ##########################################################################

    def update_target_network(self):
        """
        Transferring the estimator weights to the target weights
        """
        if self.episodes % self.target_update_period == 0:
            print_ts("About to update")
            self.DQN.update_target_net()
        self.reset()

    def reset(self):
        """
        Resetting the agent --> More explanation
        """
        super(GridAgent, self).reset()
        self.timesteps = 0
        self.episode_reward_env = 0
        self.episode_reward_shaped = 0
