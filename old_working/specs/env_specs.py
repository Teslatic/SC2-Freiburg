ACTION_TYPE = 'grid'  # compass, grid, original
MODE = 'supervised'

# GRID_DIM_X x GRID_DIM_Y
GRID_DIM_X = 20
GRID_DIM_Y = 20

REWARD_TYPE = 'distance'  # diff, distance, sparse

EPISODES = 50000
TEST_EPISODES = 50
GAMESTEPS = 0  # 0 = unlimited game time, None = map default
REPLAY_DIR = None
SAVE_REPLAY = False
STEP_MUL = 10  # 16 = 1s game time, None = map default

LOGGING = False  # Logs information in files for tmux sessions
SILENTMODE = False  # True: Just a minimum of console output
VISUALIZE = True
TEST_VISUALIZE = False

mv2beacon_specs = {
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
