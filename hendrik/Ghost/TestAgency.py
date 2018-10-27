"""
Hendrik Vloet
Copyright (C) 2018 Hendrik Vloet
Public Domain
"""
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
import os
import pandas as pd
from pathlib import Path

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


## @pacakges TestAgency
# contains the Test Agent

class TestAgent(base_agent.BaseAgent):
    ## Constructor
    def __init__(self,architecture, FLAGS, device="cpu"):
        super(TestAgent, self).__init__()
        ## where to save model and reward csv
        self._path = FLAGS.path
        ## Number of epochs to test
        self._epochs = FLAGS.epochs
        ## Set to true to see actual game screen
        self._visualize = FLAGS.visualize
        ## Train on GPU  or CPU, default is GPU if available
        self._device = device
        ## Name of the scneario/map
        self._map_name = FLAGS.map_name
        ## How many game steps per agent step
        self._step_multiplier = FLAGS.step_multiplier
        ## total reward achieved in training (pysc2 reward)
        self.total_reward = 0
        ## array to kreep track of the reward per epoch
        self.r_per_epoch = []
        ## list to keep track of the cumulative score
        self.list_score_cumulative = []
        ## actions available to the agent, chosen by design
        self._SMART_ACTIONS = [
            actions.FUNCTIONS.select_army.id,
            actions.FUNCTIONS.Move_screen.id ]
        ## setting up the network, target network and optimizer
        self._net  =  self._build_model(architecture)
        ## initializing the pysc2 environment in order to play SC2
        self._env = self._build_env()
        ## initializing the x,y coordinate pairs available to the agent
        self._xy_pairs = self._discretize_xy_grid(FLAGS.xy_grid)

        print("Network: \n{}".format(self._net))
        pytorch_total_params = sum(p.numel() for p in self._net.parameters())
        print("Total parameters: {}".format(pytorch_total_params))

    ## construct the network, the target network and the optimizer
    #
    #  @param[in] architecture: model object from architectures, chosen in main
    #  @param[out] net: feed forward neural network
    def _build_model(self, architecture):
        net = architecture.to(self._device)
        net.load_state_dict(torch.load(self._path + "model.pt", map_location=self._device))
        net.eval()

        return net


    ## construct the pysc2 environment in order to make the SC2 engine
    #  playable by the agent
    #
    #  @param[out] env: pysc2 environment object
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
            realtime = True,
            game_steps_per_episode = 0,
            visualize = self._visualize)

        return env

    ## logs rewards per epochs and cumulative reward in a csv file
    def log_reward(self):
        r_per_epoch_save_path = self._path + "reward_per_epoch.csv"

        d = {"reward per epoch" : self.r_per_epoch,
             "cumulative reward": self.list_score_cumulative}
        df = pd.DataFrame(data=d)
        with open(r_per_epoch_save_path, "w") as f:
            df.to_csv(f, header=True, index=False)


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

    def choose_action(self):
        with torch.no_grad():
            self.action_q_values = self._net(self.state)
            self.action_idx = torch.tensor([np.argmax(self.action_q_values)],
                                           dtype=torch.long, device=self._device)
            self.action_xy = self._xy_pairs[self.action_idx]
            self.x_coord = self.action_xy[0]
            self.y_coord = self.action_xy[1]


            # first action is always to select the army
            if self.actual_obs.first()==True:
                ## the action id of the pysc2 action, necessary to extract the real action
                self.pysc_action = self._SMART_ACTIONS[0]
            else:
                self.pysc_action = self._SMART_ACTIONS[1]

            self.extract_action()

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


    def test(self):
        for self.e in range(1, self._epochs):
            try:
                # get first observation
                observation = self._env.reset()
                ## verbose state observation
                self.actual_obs = observation[0]
                while (True):
                    ## state tensor
                    self.state = torch.tensor([self.actual_obs.observation.feature_screen.player_relative],
                                              dtype=torch.float, device=self._device, requires_grad = True).unsqueeze(1)

                    # agent determines action to take
                    self.step()
                    ## next_obs is the next state but densely encoded by the pysc2 engine
                    self.next_obs = self._env.step(self.action)

                    # state <- next state
                    self.actual_obs = self.next_obs[0]



                    # if in terminal state, break
                    if self.next_obs[0].last()==True:
                        break

                # reward book keeping
                self.r_per_epoch.append(self.actual_obs.observation["score_cumulative"][0])
                self.list_score_cumulative.append(self.total_reward)

                # update target network weights every _target_update epochs
                # also save the model state dictionary
                if self.e % 1 == 0:
                    self.log_reward()

            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, saving model and data!")
                self.log_reward()
                break
