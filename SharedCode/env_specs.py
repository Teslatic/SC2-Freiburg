from pysc2.env import sc2_env
from os import path
import sys

EPISODES = 10000

STEP_MULTIPLIER = 4  # 16 = 1s game time, None = map default
GAMESTEPS = 0  # 0 = unlimited game time, None = map default
VISUALIZE = True
SILENTMODE = False # True: Just a minimum of console output
LOGGING = False  # Logs information in files for tmux sessions

REWARD_TYPE = 'distance'
ACTION_TYPE = 'compass'
GRID_FACTOR = 10

# TODO: make environment specs as class, such that each map gets its own
# specification object

mv2beacon_specs = {
    'EPISODES': EPISODES,
    'STEP_MUL': STEP_MULTIPLIER,  # Standard 16
    'GAMESTEPS': GAMESTEPS,
    'VISUALIZE': VISUALIZE,
    'SAVE_REPLAY': False,
    'REPLAY_DIR': None,
    'REWARD_TYPE': REWARD_TYPE,
    'ACTION_TYPE': ACTION_TYPE,
    'GRID_FACTOR': GRID_FACTOR
}





