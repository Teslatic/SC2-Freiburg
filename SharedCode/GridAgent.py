
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

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

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
        self.DQN = DQN_module(self.batch_size,
                              self.gamma,
                              self.history_length,
                              self.size_replaybuffer,
                              self.optim_learning_rate)
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
        return self.action



    def supervised_action(self):

        # self.beacon = np.mean(self._xy_locs(
                # self.actual_obs.observation.feature_screen.player_relative == 3),
                             # axis=0).round()

            self.x_coord = self.beacon[0]
            self.y_coord = self.beacon[1]

            # print("Beacon target: {},{}".format(self.x_coord, self.y_coord))
            # print("xy space: {}".format(self._xy_pairs))

            distances = []
            for xy_pair in self._xy_pairs:
                dx = np.abs(xy_pair[0] - self.beacon[0])
                dy = np.abs(xy_pair[1] - self.beacon[1])
                distances.append(np.sqrt(dx**2 + dy**2).round())

            closest_pair = np.argmin(distances)
            # print("Closes pair: {}".format(closest_pair))
            self.action_idx = torch.tensor([closest_pair],
                                           dtype=torch.long, device=self._device)

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
            chosen_action = self.translate_to_PYSC2_action(SMART_ACTIONS[action_idx])
        else:
            action_q_values = self.DQN.predict_q_values(self.state)

            # Beste Action bestimmen
            best_action_numpy = action_q_values.detach().numpy()
            action_idx = np.argmax(best_action_numpy)
            best_action = SMART_ACTIONS[action_idx]

            chosen_action = self.translate_to_PYSC2_action(best_action)
        # square brackets around chosen_action needed for internal pysc2 state machine
        return [chosen_action], action_idx

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
    def _discretize_xy_grid(self, factor):
        """ "discretizing" action coordinates in order to keep action space small """
        x_space = np.linspace(0, 83, factor, dtype = int)
        y_space = np.linspace(0, 63, factor, dtype = int)
        xy_space = np.transpose([np.tile(x_space, len(y_space)),
                                   np.repeat(y_space, len(x_space))])

        return xy_space

    ## determins if the next action is goint to be random or greedy
    def decide(self):
        if len(self.memory) >= self._batch_size:
            self.epsilon = self._EPS_END + (self._EPS_START - self._EPS_END) \
                * np.exp(-1. * self._steps_done / self._eps_decay)
            self._steps_done += 1
            self.choice = np.random.choice(['random', 'greedy'],
                                           p=[self.epsilon, 1-self.epsilon])
        else:
            self.choice = 'random'

    ## chooses the next action
    #
    #  depends of the outcom of decide():
    #  - random: from the list of _xy_pairs, one pair is chosen and applied to the
    #  move screeen action
    #  - greedy: current state is feedforwarded through the network and a x,y pair
    #  is picked according to the max q value
    def choose_action(self):
        if self.e % self._imitation_phase_length == 0:
            self._imitation_phase = False

        if self._imitation_phase==False:
            self.decide()
            if self.choice == 'random':
                ## the index of the x,y pair that is chosen
                self.action_idx = torch.tensor([random.randint(0, len(self._xy_pairs) - 1)],
                                           dtype=torch.long, device=self._device)
                ## the actual x,y pair which will be used in the next action
                self.action_xy = self._xy_pairs[self.action_idx]
            else:
                with torch.no_grad():
                    self.action_q_values = self._net(self.state)
                    self.action_idx = torch.tensor([np.argmax(self.action_q_values)],
                                           dtype=torch.long, device=self._device)
                    self.action_xy = self._xy_pairs[self.action_idx]
            self.x_coord = self.action_xy[0]
            self.y_coord = self.action_xy[1]
            self.list_x.append(self.x_coord)
            self.list_y.append(self.y_coord)
        else:
            self.beacon = np.mean(self._xy_locs(
                self.actual_obs.observation.feature_screen.player_relative == 3),
                             axis=0).round()

            self.x_coord = self.beacon[0]
            self.y_coord = self.beacon[1]

            # print("Beacon target: {},{}".format(self.x_coord, self.y_coord))
            # print("xy space: {}".format(self._xy_pairs))

            distances = []
            for xy_pair in self._xy_pairs:
                dx = np.abs(xy_pair[0] - self.beacon[0])
                dy = np.abs(xy_pair[1] - self.beacon[1])
                distances.append(np.sqrt(dx**2 + dy**2).round())

            closest_pair = np.argmin(distances)
            # print("Closes pair: {}".format(closest_pair))
            self.action_idx = torch.tensor([closest_pair],
                                           dtype=torch.long, device=self._device)


        # first action is always to select the army
        if self.actual_obs.first()==True:
            ## the action id of the pysc2 action, necessary to extract the real action
            self.pysc_action = self._SMART_ACTIONS[0]
        else:
            self.pysc_action = self._SMART_ACTIONS[1]

        self.extract_action()


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


    ## invokes one agent step by starting the pipeline to acquire the next action
    def step(self):
        self.choose_action()


    ## calculate a pseudo reward in order to handle sparse rewards
    #
    #  the actual reward of the pysc2 engine is very sparse. 0 for every step and 1
    #  for the specific scenario reward, e.g. reaching a beacon. In order to make it
    #  easy for the agent, a pseudo reward is used. This reward is antiproportional to
    #  the distance from marine to beacon. Additionally, if the marine reaches the beacon
    #  a +10 is added to the reward.
    #
    #  This is only an intermediate step in order to find good first hyperparameters
    #  @param beacon: x,y coordinates of the current beacon
    #  @param marine_x: current x positon of the marine
    #  @param marine_y: current y position of the marine
    def calc_pseudo_reward(self):
            self.beacon = np.mean(self._xy_locs(
                self.actual_obs.observation.feature_screen.player_relative == 3),
                             axis=0).round()
            b_x, b_y = self.beacon
            if self.actual_obs.first()==False:
                try:
                    marine = self._xy_locs(self.actual_obs.observation.feature_screen.selected == 1)
                    marine = np.mean(marine, axis = 0).round()
                    self.marine_x = marine[0]
                    self.marine_y = marine[1]

                    dx = self.marine_x - b_x
                    dy = self.marine_y - b_y

                    distance = np.sqrt(dx**2 + dy**2).round(4)
                    scaling = lambda x : (x - 0)/(100 - 0)
                    # self.pseudo_reward = (0.1 * (1 - scaling(distance))).round(3)
                    self.pseudo_reward = -1 * scaling(distance).round(4)

                    #if self.actual_obs.reward==1:
                    if self.next_obs[0].reward == 1:
                        self.reward = torch.tensor([10] , device=self._device,
                                       requires_grad=False, dtype=torch.float)
                        self.pseudo_reward = 10
                    else:
                        self.reward = torch.tensor([self.pseudo_reward] , device=self._device,
                                                   requires_grad=False, dtype=torch.float)
                except:
                    pass
            else:
                self.marine_x = -1
                self.marine_y = -1
                self.reward = torch.tensor([0] , device=self._device,
                                       requires_grad=False, dtype=torch.float)
                self.pseudo_reward = 0

            self.pseudo_reward_per_epoch += self.pseudo_reward
            self.total_reward += self.actual_obs.reward


    ## optimizes the network and updates the target network every _target_net_udpdate episodes
    #
    #  In this method, a batch is sampled from the ReplayBuffer memory and organized in its
    #  transition chunks:
    #  - state_batch: batch of states
    #  - action_batch: batch of actions
    #  - reward_batch: batch of rewards
    #  - next_states: batch of next states
    #
    #  The optimize method takes the batches and computes q values with the actual net and
    #  q values for the next states via the target network. Afterwards, the td target is calculated
    #  and used to get the loss between the actual q values and the td target
    ##  @param[out] loss: current loss value computed by the opzimizer
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
        reward_batch = torch.cat(batch.reward).detach()

        q_action = self._net(state_batch).gather(1, action_batch)

        q_action_next = torch.zeros(self._batch_size, device=self._device)
        q_action_next[non_final_mask] = self._target_net(non_final_next_states).max(1)[0].detach()

        # non_final_next_action_q  = non_final_next_action_q.max(1)[0].detach()


        td_target_actions = (q_action_next * self._gamma) + reward_batch

        loss = F.mse_loss(q_action, td_target_actions.unsqueeze(1))

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss


    ## prints some status infos, e.g. reward, episode, etc.
    def print_status(self):

        print("Episode: {}\t Total pysc2 reward: {}\t Pseudo Reward: {}".format(
            self.e,self.total_reward,self.reward))
        print("Epsilon: {:.4f}".format(self.epsilon))
        print("Beacon at {}".format(self.beacon))
        print("Marine at {}".format((self.marine_x, self.marine_y)))
        print("Memory length: {}".format(len(self.memory)))
        try:
            print("Loss: {}".format(self.loss))
        except:
             pass
        print(self.action)
        print(100 * "=")


    ## training loop
    #
    #  the play method is the actual training loop
    ## @param state: tensor, where the current screen state is saved (player relative at the moment). The state has the dimensions (batch_size x 1 x 84 x 84)
    ## @param next_state: same as state but for the next state
    ## @param reward: tensor, containting the current reward
    def play(self):
        for self.e in range(1, self._epochs):
            self.pseudo_reward_per_epoch = 0
            try:
                # get first observation
                observation = self._env.reset()
                ## verbose state observation
                self.actual_obs = observation[0]
                while (True):
                    ## state tensor
                    self.state = torch.tensor([self.actual_obs.observation.feature_screen.player_relative],
                                              dtype=torch.float, device=self._device, requires_grad = False).unsqueeze(1)

                    # agent determines action to take
                    self.step()
                    ## next_obs is the next state but densely encoded by the pysc2 engine
                    self.next_obs = self._env.step(self.action)
                    # get pseudo reward
                    self.calc_pseudo_reward()

                    # if in last step, write only None in next state
                    if self.next_obs[0].last():
                        ## next_state is the more verbose version of next_obs and also used by the network
                        self.next_state = None
                    else:
                        self.next_state = torch.tensor([self.next_obs[0].observation.feature_screen.player_relative],\
                                                       dtype=torch.float, device=self._device,
                                                       requires_grad = False).unsqueeze(1)

                    # save transition
                    self.memory.push(self.state, self.action_idx, self.reward , self.next_state,
                                     self.next_obs[0].step_type.value)

                    # state <- next state
                    self.actual_obs = self.next_obs[0]

                    # train/optimize
                    if self._imitation_phase==False:
                        self.loss = self.optimize()

                    # print status message
                    self.print_status()

                    # if in terminal state, break
                    if self.next_obs[0].last()==True:
                        break

                # reward book keeping
                # if self._imitation_phase==False:
                self.r_per_epoch.append(self.actual_obs.observation["score_cumulative"][0])
                self.list_score_cumulative.append(self.total_reward)
                self.list_pseudo_reward_per_epoch.append(self.pseudo_reward_per_epoch)
                self.list_epsilon.append(self.epsilon)
                # self.print_status()
                # update target network weights every _target_update epochs
                # also save the model state dictionary
                if self.e % self._target_update == 0:
                    self._target_net.load_state_dict(self._net.state_dict())
                    self._save_model()
                    self.log_reward()
                    self.log_coordinates()
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, saving model and data!")
                self._save_model()
                self.log_reward()
                break




