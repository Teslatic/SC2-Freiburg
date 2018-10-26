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

## @package Agency
#  The agency file contains all agents and tools the agents use in order to play
#  the game.
#  - ReplayBuffer
#  - BaseAgent

## Documentation for ReplayBuffer
#
#  The replay buffer saves transition tuples via the push method and samples a
#  random batch out of its memory (uniform distribution)
class ReplayBuffer(object):
    ## Constructor
    #
    # @param capacity describes the memory size
    # @param memory a list where the samples are saved
    # @param position index helper
    # @param Transition named tuple for transiton contents: [state, action, reward, next state, step type]
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
            ('state', 'action', 'reward', 'next_state',
             'step_type'))

    ##  returns current number of saved transitions
    def __len__(self):
        return len(self.memory)

    ## get transtion by index
    def __getitem__(self, key):
        return self.memory[key]

    ## saves one transtion in the replay buffer/stack
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position+1) % self.capacity

    ##  return random sample transiton from experience
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

## Documentation for BaseAgent
#
#  This is the central agent which controls all program flow comprising:
#  - parsing flags to hyperparameter member variables
#  - building the net, target net and optimizer
#  - setting up the pysc2 environment in order to use it
#  - playing
#  - training
#  - optimizing
class BaseAgent(base_agent.BaseAgent):
    ## Constructor
    def __init__(self,architecture, FLAGS, device="cpu"):
        super(BaseAgent, self).__init__()
        ## name of the experiment
        self._name = FLAGS.name
        ## where to save model and reward csv
        self._path = FLAGS.path
        ## Learning rate for the optimizer
        self._lr = FLAGS.learning_rate
        ## Discount factor
        self._gamma = FLAGS.learning_rate
        ## Batch size
        self._batch_size = FLAGS.batch_size
        ## Update the target network every N epochs
        self._target_update = FLAGS.target_update
        ## Number of epochs to train
        self._epochs = FLAGS.epochs
        ## Replay Buffer capacity
        self._memory_size = FLAGS.memory_size
        ## Set to true to see actual game screen
        self._visualize = FLAGS.visualize
        ## Train on GPU  or CPU, default is GPU if available
        self._device = device
        ## Name of the scneario/map
        self._map_name = FLAGS.map_name
        ## How many game steps per agent step
        self._step_multiplier = FLAGS.step_multiplier
        ## Epsilon decay rate; high -> slow decay
        self._eps_decay = FLAGS.epsilon
        ## epsilon, has to be initialized with 1.0
        self.epsilon = 1.0
        ## start value for epsilon decay
        self._EPS_START = 1.0
        ## end value for epsilon decay
        self._EPS_END = 0.1
        ## step counter for epsilon calculation
        self._steps_done = 0
        ## displays choice: random or greed, init with None
        self._choice = None
        ## total reward achieved in training (pysc2 reward)
        self.total_reward = 0
        ## array to kreep track of the reward per epoch
        self.r_per_epoch = []
        ## list to keep track of the cumulative score
        self.list_score_cumulative = []
        ## list to keep track of the pseudo reward per epoch
        self.list_pseudo_reward_per_epoch = []
        ## actions available to the agent, chosen by design
        self._SMART_ACTIONS = [
            actions.FUNCTIONS.select_army.id,
            actions.FUNCTIONS.Move_screen.id ]
        ## initializing of the ReplayBuffer
        self.memory = ReplayBuffer(self._memory_size)
        ## setting up the network, target network and optimizer
        self._net, self._target_net, self._optimizer = \
            self._build_model(architecture)
        ## initializing the pysc2 environment in order to play SC2
        self._env = self._build_env()
        ## initializing the x,y coordinate pairs available to the agent
        self._xy_pairs = self._discretize_xy_grid(FLAGS.xy_grid)
        ## imitation phase length
        self._imitation_phase_length = FLAGS.imitation_length
        self._imitation_phase = True
        ## list of chosen x coordinates
        self.list_x = []
        ## list of chosen y coordinates
        self.list_y = []

        print("Network: \n{}".format(self._net))
        print("Optimizer: \n{}".format(self._optimizer))
        print("Target Network: \n{}".format(self._target_net))
        pytorch_total_params = sum(p.numel() for p in self._net.parameters())
        print("Total parameters: {}".format(pytorch_total_params))

    ## construct the network, the target network and the optimizer
    #
    #  @param[in] architecture: model object from architectures, chosen in main
    #  @param[out] net: feed forward neural network
    #  @param[out] target_net: feed forward neural target network
    #  @param[out] optimizer: Optimizer of choice, default is Adam
    def _build_model(self, architecture):
        net = architecture.to(self._device)
        target_net = architecture.to(self._device)
        optimizer = optim.Adam(net.parameters(), lr=self._lr)

        return net, target_net, optimizer


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
            realtime = False,
            game_steps_per_episode = 0,
            visualize = self._visualize)

        return env


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


    ## returns the xy location of a given game-object, like a marine for example
    #
    #  @param[in] mask: Mask should be a set of bools from comparison with a feature layer
    #  @param[out] list(zip(x,y)): list of x,y coordinates where the object is located
    def _xy_locs(self, mask):
        y, x = mask.nonzero()

        return list(zip(x, y))


    ## saves the pytorch model
    def _save_model(self):
        save_path = self._path + "/experiment/" + self._name + "/model/model.pt"
        Path((self._path + "/experiment/" + self._name + "/model")).mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), save_path)


    ## logs rewards per epochs and cumulative reward in a csv file
    def log_reward(self):
        r_per_epoch_save_path = self._path  + "/experiment/" + self._name + "/csv/reward_per_epoch.csv"
        Path((self._path + "/experiment/" + self._name + "/csv")).mkdir(parents=True, exist_ok=True)

        d = {"sparse reward per epoch" : self.r_per_epoch,
             "sparse cumulative reward": self.list_score_cumulative,
             "pseudo reward per epoch" : self.list_pseudo_reward_per_epoch }
        df = pd.DataFrame(data=d)
        with open(r_per_epoch_save_path, "w") as f:
            df.to_csv(f, header=True, index=False)


    def log_coordinates(self):
        save_path = self._path + "/experiment/" + self._name + "/csv/coordinates.csv"
        Path((self._path + "/experiment/" + self._name + "/csv")).mkdir(parents=True, exist_ok=True)
        d = {"x" : self.list_x,
             "y":  self.list_y}
        df = pd.DataFrame(data=d)
        with open(save_path, "w") as f:
            df.to_csv(f, header=True, index=False)


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
                                       requires_grad=True, dtype=torch.float)
                        self.pseudo_reward = 10
                    else:
                        self.reward = torch.tensor([self.pseudo_reward] , device=self._device,
                                                   requires_grad=True, dtype=torch.float)
                except:
                    pass
            else:
                self.marine_x = -1
                self.marine_y = -1
                self.reward = torch.tensor([0] , device=self._device,
                                       requires_grad=True, dtype=torch.float)
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
                                              dtype=torch.float, device=self._device, requires_grad = True).unsqueeze(1)

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
                                                       requires_grad = True).unsqueeze(1)

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

    # def pendulum_poc(self):




################################################################################
# TEST AGENT! 
#
################################################################################

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
