from pysc2.env import sc2_env
from os import path
import sys

BASE_AGENT = 'base_agent'
GAMMA = 0.99
OPTIM_LR = 0.001
BATCH_SIZE = 32
TARGET_UPDATE_PERIOD = 5
HIST_LENGTH = 1
REPLAY_SIZE = 30000
SUPERVISED_EPISODES = 2
DEVICE = 'cpu' # will be overwritten by main
# DEVICE = "cuda:0"

epsilon_specs = {
                'EPSILON': 0.02,
                'EPS_START': 0.02,
                'EPS_END': 0.01,
                'EPS_DECAY': 10000 # 50000
                }

mv2beacon_specs = {
              'TYPE': BASE_AGENT,  # Standard: BASE_AGENT
              'GAMMA': GAMMA,  # Standard 0.99
              'OPTIM_LR': OPTIM_LR,  # Standard 0.0001
              'BATCH_SIZE': BATCH_SIZE,  # Standard 32
              'TARGET_UPDATE_PERIOD': TARGET_UPDATE_PERIOD,   # Standard 5
              'HIST_LENGTH': HIST_LENGTH,   # Standard 4
              'REPLAY_SIZE': REPLAY_SIZE,
              'DEVICE': DEVICE,
              'EPSILON_FILE': epsilon_specs,
              'EXP_PATH': None
              }

