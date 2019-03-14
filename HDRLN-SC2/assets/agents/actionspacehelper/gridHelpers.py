# python imports
import numpy as np
import random

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

# ##########################################################################
# Action Selection
# ##########################################################################

def retrieve_available_actions(available_actions):
    """
    """
    

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
