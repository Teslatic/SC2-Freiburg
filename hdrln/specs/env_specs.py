ENV_ID = 'collectmineralshards'
# ENV_ID = 'gym-sc2-mineralshards-v0'
# ENV_ID = 'gym-sc2-defeatroaches-v0'

ACTION_TYPE = 'grid'  # compass, grid, pysc2
MODE = 'learning'

# GRID_DIM_X x GRID_DIM_Y
GRID_DIM_X = 20
GRID_DIM_Y = 20

REWARD_TYPE = 'sparse'  # diff, distance, sparse, original(collectmineralshards)

EPISODES = 40000
TEST_EPISODES = 5
GAMESTEPS = 0  # 0 = unlimited game time, None = map default
REPLAY_DIR = None
SAVE_REPLAY = False
STEP_MUL = 16  # 16 = 1s game time, None = map default

LOGGING = False  # Logs information in files for tmux sessions
SILENTMODE = False  # True: Just a minimum of console output
VISUALIZE = False
TEST_VISUALIZE = True

env_specs = {
    'ENV_ID': ENV_ID,
    'ACTION_TYPE': ACTION_TYPE,
    'GRID_DIM_X': GRID_DIM_X,
    'GRID_DIM_Y': GRID_DIM_Y,

    'REWARD_TYPE': REWARD_TYPE,

    'EPISODES': EPISODES,
    'GAMESTEPS': GAMESTEPS,
    'REPLAY_DIR': REPLAY_DIR,
    'SAVE_REPLAY': SAVE_REPLAY,
    'STEP_MUL': STEP_MUL,  # Standard 16 = 1s/action
    'MODE': MODE,
    'TEST_EPISODES': TEST_EPISODES,

    'LOGGING': LOGGING,
    'SILENTMODE': SILENTMODE,
    'VISUALIZE': VISUALIZE,
    'TEST_VISUALIZE': TEST_VISUALIZE}

# TODO: make environment specs as class, such that each map gets its own
# specification object
