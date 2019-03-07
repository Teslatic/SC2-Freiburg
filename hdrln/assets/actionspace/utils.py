# python imports
import numpy as np
import random
from assets.actionspace.smart_actions import SMART_ACTIONS_COMPASS
from assets.actionspace.smart_actions import SMART_ACTIONS_GRID
from assets.actionspace.smart_actions import ALL_ACTIONS_PYSC2
from assets.actionspace.smart_actions import ALL_ACTIONS_PYSC2_IDX
from assets.actionspace.smart_actions import PYSC2_to_Qvalueindex

def discretize_xy_grid(grid_dim_x, grid_dim_y):
    """
    Discretizing action coordinates in order to keep action space small
    """
    x_space = np.linspace(0, 83, grid_dim_x, dtype=int)
    y_space = np.linspace(0, 63, grid_dim_y, dtype=int)
    xy_space = np.transpose([np.tile(x_space, len(y_space)),
                            np.repeat(y_space, len(x_space))])
    dim_actions = len(xy_space)
    smart_actions = range(0, dim_actions)
    return xy_space, dim_actions, smart_actions

def setup_action_space(agent_specs):
    """
    Returns action space and action dimensionality
    """
    action_type = agent_specs["ACTION_TYPE"]
    if action_type == 'compass':
        action_space = SMART_ACTIONS_COMPASS
        dim_actions = len(action_space)

    elif action_type == 'grid':
        grid_dim_x = agent_specs['GRID_DIM_X']
        grid_dim_y = agent_specs['GRID_DIM_Y']
        xy_space, dim_actions, action_space = discretize_xy_grid(grid_dim_x, grid_dim_y)

    elif action_type == 'pysc2':
        action_space = ALL_ACTIONS_PYSC2
        dim_actions = len(action_space)

    elif action_type == 'minigame':
        raise("This action space type has not been implemeted yet.")
        exit()

    elif action_type == 'original':
        raise("This action space type has not been implemeted yet.")
        exit()
    else:
        raise("Action type " + action_type + " is unknown!")
        exit()

    return action_space, dim_actions

# ##########################################################################
# Action Selection
# ##########################################################################

def inject_noise(self, closest_pair):
    """
    Injects noise on the supervised actions to make them sub optimal.
    """

    noise_scalar = random.randint(-NOISE_BOUND, NOISE_BOUND)
    noise_factor = random.randint(-NOISE_BOUND, NOISE_BOUND)
    noise_x = noise_scalar
    noise_y = noise_scalar + noise_factor * random.choice(
                                                        [-grid_dim_x,
                                                         +grid_dim_x])
    which_noise = random.choice([noise_x, noise_y])
    # -1 for boundary case
    return max(0, min(len(_xy_pairs) - 1, closest_pair + which_noise))


def supervised_action(self):
    """
    This method selects a grid point which is the closest to the beacon.
    """

    dx = _xy_pairs[:,0] - beacon_center[0]
    dy = _xy_pairs[:,1] - beacon_center[1]
    distances = np.sqrt(dx**2 + dy**2).round().astype(int)

    # TODO: Check if correct
    closest_pair = np.argmin(distances)

    action_idx = inject_noise(closest_pair)
    action = action_idx
    return action, action_idx
