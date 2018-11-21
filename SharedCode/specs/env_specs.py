ACTION_TYPE = 'grid'  # compass, grid, original
GRID_FACTOR = 20
REWARD_TYPE = 'distance'  # diff, distance, sparse

EPISODES = 5000
GAMESTEPS = 0  # 0 = unlimited game time, None = map default
REPLAY_DIR = None
SAVE_REPLAY = False
STEP_MUL = 32  # 16 = 1s game time, None = map default

LOGGING = False  # Logs information in files for tmux sessions
SILENTMODE = False  # True: Just a minimum of console output
VISUALIZE = False

mv2beacon_specs = {
    'ACTION_TYPE': ACTION_TYPE,
    'GRID_FACTOR': GRID_FACTOR,
    'REWARD_TYPE': REWARD_TYPE,

    'EPISODES': EPISODES,
    'GAMESTEPS': GAMESTEPS,
    'REPLAY_DIR': REPLAY_DIR,
    'SAVE_REPLAY': SAVE_REPLAY,
    'STEP_MUL': STEP_MUL,  # Standard 16 = 1s/action

    'LOGGING': LOGGING,
    'SILENTMODE': SILENTMODE,
    'VISUALIZE': VISUALIZE}

# TODO: make environment specs as class, such that each map gets its own
# specification object
