from pysc2.env import sc2_env

EPISODES = 10000
TEST_EPISODES = 5
BASE_AGENT = 'base_agent'
SCREEN_DIM = 84
MINIMAP_DIM = 64
GAMMA = 1.0
OPTIM_LR = 0.0001
BATCH_SIZE = 32
TARGET_UPDATE_PERIOD = 10
HIST_LENGTH = 1
REPLAY_SIZE = 100000
DEVICE = 'cpu'

MAP = 'MoveToBeacon'
PLAYERS = [sc2_env.Agent(sc2_env.Race.terran)]
STEP_MULTIPLIER = 16  # 16 = 1s game time, None = map default
GAMESTEPS = None  # 0 = unlimited game time, None = map default
VISUALIZE = False

epsilon_file = {
                'EPSILON': 1.0,
                'EPS_START': 1.0,
                'EPS_END': 0.1,
                'EPS_DECAY': 20000
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
              'EPSILON_FILE': epsilon_file
              }

env_file = {
            'EPISODES': EPISODES,
            'MAP_NAME': MAP,
            'PLAYERS': PLAYERS,
            'STEP_MULTIPLIER': STEP_MULTIPLIER,  # Standard 16
            'GAMESTEPS': GAMESTEPS,
            'VISUALIZE': VISUALIZE
            }

test_env_file = {
            'EPISODES': TEST_EPISODES,
            'MAP_NAME': MAP,
            'PLAYERS': PLAYERS,
            'STEP_MULTIPLIER': STEP_MULTIPLIER,  # Standard 16
            'GAMESTEPS': GAMESTEPS,
            'VISUALIZE': False
            }
