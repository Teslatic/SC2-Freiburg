#!/usr/bin/env python3

'''
helpers
'''
def setup_agent():
  agent = BaseAgent(screen_dim = SCREEN_DIM, minimap_dim = MINIMAP_DIM,
          batch_size=BATCH_SIZE, target_update_period=TARGET_UPDATE_PERIOD,
          history_length=HIST_LENGTH)
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

'''
custom classes
'''


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
    self.position = (self.position +1) % self.capacity

  def sample(self, batch_size):
    ''' return random sample transition from ER '''
    return random.sample(self.memory, batch_size)

class History(object):
  '''
 stacks frames as some kind of histor for the agent since training data is
  not i.i.d
  '''
  def __init__(self, HIS_LENGTH):
    # self.history = np.zeros((HIS_LENGTH, 84, 84))
    self.history = np.zeros((1,HIS_LENGTH,84,84))
    self.capacity = HIS_LENGTH

  def __getitem__(self, key):
    return self.history[key]

  def stack(self, state):
    tmp_state = state.unsqueeze(1).numpy()
    self.history = np.delete(self.history,0,1)
    self.history = np.append(self.history,tmp_state,1)

    return torch.tensor(self.history,dtype=torch.float)

'''
main function

'''
def main(unused_argv):

  '''
  function helper for status printing
  '''
  def print_status(agent, env):
    print("Epsilon: {:.2f}\t| choice: {}".format(agent.epsilon,agent.choice))
    print("Episode {}\t| Step {}\t| Total Steps: {}".format(agent.episodes, agent.timesteps, agent.steps))
    print("Chosen action: {}".format(action))
    print("chosen coordinates [x,y]: {}".format((x_coord, y_coord)))
    print("Beacon center location [x,y]: {}".format(beacon_center))
    print("Current Episode Score: {}\t| Total Score: {}".format(env._last_score[0],agent.reward))
    print("Loss: {:.5f}".format(loss))
    print("{}".format(agent.update_status))
    print("----------------------------------------------------------------")

    return
  torch.set_printoptions(linewidth=750, profile="full")

  agent, players, agent_interface = setup_agent()
  memory = ReplayBuffer(1000000)
  hist = History(HIST_LENGTH)
  try:
    while True:
      with  setup_env(agent,players,agent_interface) as env:
        agent.setup(env.observation_spec(), env.action_spec())
        internal_state = env.reset()
        agent.reset()
        print("----------------------------------------------------------------")
        while True:
          # get beacon center for debug reference
          beacon_center = np.mean(_xy_locs(internal_state[0].observation["feature_screen"]["player_relative"] == 3),axis=0).round()

          # print status
          try:
            print_status(agent, env)
          except:
            pass

          # extraction of the transition tuple we are interested in from the internal state machine
          state = torch.tensor([internal_state[0].observation["feature_screen"]["player_relative"]],dtype=torch.float)

          # put current state on history stack
          state_history_tensor = hist.stack(state)

          # make one step
          internal_next_state, action, action_idx, x_coord, y_coord = agent.step(internal_state[0], env, state_history_tensor)

          # collect transition data
          reward = torch.tensor([internal_state[0].reward], device=device,dtype=torch.float)
          step_type = torch.tensor([internal_state[0].step_type], device=device,dtype=torch.int)
          next_state = torch.tensor([internal_next_state[0].observation["feature_screen"]["player_relative"]],dtype=torch.float)

          # push next state on next state history stack
          next_state_history_tensor = hist.stack(next_state)

          # save transition tuple to the memory buffer
          memory.push(state_history_tensor, action_idx, x_coord, y_coord, reward, next_state_history_tensor, step_type)


          if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)

            # zipping necessary for reasons i dont really understand
            batch = Transition(*zip(*transitions))

            # loss is not very expressive, but included just for sake of completeness
            loss = agent.optimize(batch)

          # check if done, i.e. step_type==2
          if step_type==2:
            break

          # s_t <- s_(t+1), i.e old state <- new state
          internal_state = internal_next_state
  except KeyboardInterrupt:
    pass

if __name__ == "__main__":
  # pysc2 modules
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


  # custom imports
  from BaseAgent import BaseAgent, SMART_ACTIONS
  from AtariNet import DQN

  # normal python modules
  import random
  import argparse
  from collections import namedtuple
  import numpy as np

  # CONSTANTS
  SCREEN_DIM = 84
  MINIMAP_DIM = 64
  GAMESTEPS = None # 0 = unlimited game time, None = map default
  STEP_MULTIPLIER = 16 # 16 = 1s game time, None = map default
  BATCH_SIZE = 128
  # every n steps update target weights
  TARGET_UPDATE_PERIOD = 40
  VISUALIZE = True
  HIST_LENGTH = 10

  # Initalizing
  Transition = namedtuple('Transition', ('state', 'action', 'x_coord', 'y_coord', 'reward', 'next_state','step_type'))
  replay_memory_size = 100
  replay_memory = []
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  app.run(main)
