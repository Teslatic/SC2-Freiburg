from pysc2.env import sc2_env
from os import path
import sys

EPISODES = 10000
TEST_EPISODES = 1
BASE_AGENT = 'base_agent'
SCREEN_DIM = 84
MINIMAP_DIM = 64
GAMMA = 0.99
OPTIM_LR = 0.0001
BATCH_SIZE = 32
TARGET_UPDATE_PERIOD = 5
HIST_LENGTH = 1
REPLAY_SIZE = 1000000
SUPERVISED_EPISODES = 2
DEVICE = 'cpu' # will be overwritten by main
# DEVICE = "cuda:0"

MAP = 'MoveToBeacon'
PLAYERS = [sc2_env.Agent(sc2_env.Race.terran)]
STEP_MULTIPLIER = 2  # 16 = 1s game time, None = map default
EPISODES = 0  # 0 = unlimited game time, None = map default
EPISODES_TEST = 5  # 0 = unlimited game time, None = map default
VISUALIZE = True
SILENTMODE = True # True: Just a minimum of console output

epsilon_file = {
                'EPSILON': 0.20,
                'EPS_START': 0.20,
                'EPS_END': 0.01,
                'EPS_DECAY': 10000 # 50000
                }

agent_file = {
              'TYPE': BASE_AGENT,  # Standard: BASE_AGENT
              'SCREEN_DIM': SCREEN_DIM,  # Standard 84
              'MINIMAP_DIM': MINIMAP_DIM,  # Standard 64
              'GAMMA': GAMMA,  # Standard 0.99
              'OPTIM_LR': OPTIM_LR,  # Standard 0.0001
              'BATCH_SIZE': BATCH_SIZE,  # Standard 32
              'TARGET_UPDATE_PERIOD': TARGET_UPDATE_PERIOD,   # Standard 5
              'HIST_LENGTH': HIST_LENGTH,   # Standard 4
              'REPLAY_SIZE': REPLAY_SIZE,
              'DEVICE': DEVICE,
              'EPSILON_FILE': epsilon_file,
              'SILENTMODE': SILENTMODE,
              'SUPERVISED_EPISODES': SUPERVISED_EPISODES
              }

env_file = {
            'EPISODES': EPISODES,
            'MAP_NAME': MAP,
            'PLAYERS': PLAYERS,
            'STEP_MULTIPLIER': STEP_MULTIPLIER,  # Standard 16
            'GAMESTEPS': EPISODES,
            'VISUALIZE': VISUALIZE,
            'SAVE_REPLAY': False,
            'REPLAY_DIR': None
            }

test_env_file = {
            'EPISODES': TEST_EPISODES,
            'MAP_NAME': MAP,
            'PLAYERS': PLAYERS,
            'STEP_MULTIPLIER': STEP_MULTIPLIER,  # Standard 16
            'GAMESTEPS': EPISODES_TEST,
            'VISUALIZE': True,
            'SAVE_REPLAY': True,
            'REPLAY_DIR': path.dirname(path.abspath(sys.modules['__main__'].__file__))
            }
