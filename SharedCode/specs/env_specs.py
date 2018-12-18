ACTION_TYPE = 'compass'  # compass, grid, original

# GRID_DIM_X x GRID_DIM_Y
GRID_DIM_X = 15
GRID_DIM_Y = 15

REWARD_TYPE = 'diff'  # diff, distance, sparse

EPISODES = 2000
TEST_EPISODES = 10
GAMESTEPS = 0  # 0 = unlimited game time, None = map default
REPLAY_DIR = None
SAVE_REPLAY = False
STEP_MUL = 4   # 16 = 1s game time, None = map default

LOGGING = False  # Logs information in files for tmux sessions
SILENTMODE = False  # True: Just a minimum of console output
VISUALIZE = False
TEST_VISUALIZE = True

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
    'TEST_EPISODES': TEST_EPISODES,

    'LOGGING': LOGGING,
    'SILENTMODE': SILENTMODE,
    'VISUALIZE': VISUALIZE,
    'TEST_VISUALIZE': TEST_VISUALIZE}

# TODO: make environment specs as class, such that each map gets its own
# specification object
