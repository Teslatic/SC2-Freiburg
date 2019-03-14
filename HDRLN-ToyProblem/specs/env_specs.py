EPISODES = 3000
TEST_EPISODES = 200

LOGGING = False  # Logs information in files for tmux sessions
SILENTMODE = False  # True: Just a minimum of console output
VISUALIZE = False

cartpole_specs = {
    'EPISODES': EPISODES,
    'TEST_EPISODES': TEST_EPISODES,
    'LOGGING': LOGGING,
    'SILENTMODE': SILENTMODE,
    'VISUALIZE': VISUALIZE}

# TODO: make environment specs as class, such that each map gets its own
# specification object
