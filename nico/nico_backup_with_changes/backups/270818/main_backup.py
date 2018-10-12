#!/usr/bin/env python3

def main(unused_argv):

  # GLOBAL CONSTANTS
  SCREEN_DIM = 84
  MINIMAP_DIM = 64
  GAMESTEPS = None # 0 = unlimited game time, None = map default
  STEP_MULTIPLIER = 16 # 16 = 1s game time, None = map default
  # every n steps update target weights (BaseAgent)
  # every n episodes update target weights (TripleAgent)
  EPISODES = 100
  HIST_LENGTH = 4
  BATCH_SIZE = 32
  TARGET_UPDATE_PERIOD = 5

  agent_file = {
                'SCREEN_DIM': SCREEN_DIM,  # Standard 84
                'MINIMAP_DIM': MINIMAP_DIM,  # Standard 64
                'BATCH_SIZE': BATCH_SIZE,  # Standard 32
                'TARGET_UPDATE_PERIOD': TARGET_UPDATE_PERIOD,   # Standard 5
                'HIST_LENGTH': HIST_LENGTH   # Standard 4
                }

  env_file = {
              'MAP_NAME': 'MoveToBeacon',
              'PLAYERS': [sc2_env.Agent(sc2_env.Race.terran)],
              'STEP_MULTIPLIER': STEP_MULTIPLIER,  # Standard 16
              'GAMESTEPS': GAMESTEPS,
              'VISUALIZE': True
              }

  def setup_env(env_file, agent_interface):
    """
    Refer to PYSC2 documentation.
    """
    print("Initalizing environment")
    env = sc2_env.SC2Env(
        map_name = env_file['MAP_NAME'],
        players = env_file['PLAYERS'],
        agent_interface_format = agent_interface,
        step_mul = env_file['STEP_MULTIPLIER'],
        game_steps_per_episode = env_file['GAMESTEPS'],
        visualize = env_file['VISUALIZE'])
    return env

  def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))

  try:
    # Initalizing
    Transition = namedtuple('Transition', ('state', 'action', 'x_coord', 'y_coord', 'reward', 'next_state','step_type'))
    hist = History(HIST_LENGTH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(linewidth=750, profile="full")

    # Setting up the agent
    agent = TripleAgent(agent_file)
    agent_interface = agent.setup_interface()

    for _ in range(EPISODES):
      with setup_env(env_file, agent_interface) as env:
        observation = env.reset()  # Better not call this internal_state. More like observation.
        agent.setup(env.observation_spec(), env.action_spec())  # Necessary?
        agent.reset()
        while True:
          # #########################################
          # Should not contain to much loose code:
          # 1. Calculating debug info
          # 2. Performing agent step with internal calculations
          # 3. Compare debug and agent information
          # #########################################

          actual_obs = observation[0]
          # print(actual_obs)

          # get beacon center for debug reference
          beacon_center = np.mean(_xy_locs(actual_obs.observation.feature_screen.player_relative == 3),axis=0).round()

          # print status
          try:
            # This is too clean. Maybe put this in the agent as well. Put it at the end.
            agent.print_status(env, action, x_coord, y_coord, beacon_center, action_loss, x_coord_loss, y_coord_loss)
          except:
            pass

          # extraction of the transition tuple we are interested in from the internal state machine
          state = torch.tensor([actual_obs.observation.feature_screen.player_relative],dtype=torch.float)

          # put current state on history stack
          state_history_tensor = hist.stack(state)

          # make one step
          internal_next_state, action, action_idx, x_coord, y_coord = agent.step(actual_obs, env, state_history_tensor)
          next_obs = internal_next_state[0]

          # collect transition data
          reward = torch.tensor([actual_obs.reward], device=device,dtype=torch.float)
          step_type = torch.tensor([actual_obs.step_type], device=device,dtype=torch.int)
          next_state = torch.tensor([next_obs.observation.feature_screen.player_relative],dtype=torch.float)

          # push next state on next state history stack
          next_state_history_tensor = hist.stack(next_state)

          # save transition tuple to the memory buffer
          memory.push(state_history_tensor, action_idx, x_coord, y_coord, reward, next_state_history_tensor, step_type)

          ########
          # OPTIMIZE ON BATCH
          ########
          if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)

            # zipping necessary for reasons i dont really understand
            batch = Transition(*zip(*transitions))

            # loss is not very expressive, but included just for sake of completeness
            action_loss, x_coord_loss, y_coord_loss = agent.optimize(batch)

          ########
          # UPDATE TARGET NETWORKS --> Put this into agent
          ########

                # update target nets
            if agent.episodes % TARGET_UPDATE_PERIOD == 0:
              agent.update_target_net(agent.action_net, agent.action_target_net)
              agent.update_target_net(agent.x_coord_net, agent.x_coord_target_net)
              agent.update_target_net(agent.y_coord_net, agent.y_coord_target_net)
            else:
              agent.update_status = " - "

          # check if done, i.e. step_type==2
          if step_type == 2:
            break

          # s_t <- s_(t+1), i.e old state <- new state
          internal_state = internal_next_state
  except KeyboardInterrupt:
    pass

if __name__ == "__main__":
  print("Starting main")
  # pysc2 modules
  print("Importing packages")
  from pysc2.agents import  random_agent, scripted_agent
  from pysc2.env import sc2_env
  from pysc2.lib import actions, features, units
  from absl import app

  ### torch imports
  import torch
  import torch.nn as nn
  import torch.optim as optim
  import torch.nn.functional as F
  import torchvision.transforms as T

  import os
  from os import path
  import sys
  if "../" not in sys.path:
    sys.path.append("../")

  # custom imports
  from assets.agents.BaseAgent import BaseAgent, SMART_ACTIONS
  from assets.agents.TripleAgent import TripleAgent
  from assets.memory.ReplayBuffer import ReplayBuffer
  from assets.memory.History import History
  from AtariNet import DQN

  # normal python modules
  import random
  import argparse
  from collections import namedtuple
  import numpy as np

  app.run(main)
