# python imports
import numpy as np

# custom imports
from assets.RL.DQN_module import DQN_module
from assets.helperFunctions.timestamps import print_timestamp as print_ts
from assets.agents.Move2BeaconAgent import Move2BeaconAgent


class GridAgent(Move2BeaconAgent):
    """
    TODO: explain the grid stuff LUL
    This is a simple agent that uses an PyTorch DQN_module as Q value
    approximator. Current implemented features of the agent:
    - Simple initializing with the help of an agent_specs.
    - Policy switch between imitation and epsilon greedy learning session.
    - Storage of experience into simple Experience Replay Buffer
    - Intermediate saving of model weights

    To be implemented:
    - Saving the hyperparameter file of the experiments.
    - Storing the DQN model weights in case of Runtime error.
    """
    # ##########################################################################
    # Initializing the agent
    # ##########################################################################
    def __init__(self, agent_specs):
        """
        Refer to Move2BeaconAgent.
        """
        super(GridAgent, self).__init__(agent_specs)

    def setup_dqn(self):
        """
        Setting up the DQN module with the grid action space.
        """
        self._xy_pairs, self.dim_actions, self.smart_actions = \
            self.discretize_xy_grid()
        self.DQN = DQN_module(self.batch_size,
                              self.gamma,
                              self.history_length,
                              self.size_replaybuffer,
                              self.optim_learning_rate,
                              self.dim_actions)
        self.device = self.DQN.device
        print_ts("DQN module has been initalized")


    def discretize_xy_grid(self):
        """
        Discretizing action coordinates in order to keep action space small
        """
        x_space = np.linspace(0, 83, self.grid_dim_x, dtype=int)
        y_space = np.linspace(0, 63, self.grid_dim_y, dtype=int)
        xy_space = np.transpose([np.tile(x_space, len(y_space)),
                                np.repeat(y_space, len(x_space))])
        dim_actions = len(xy_space)
        smart_actions = range(0, dim_actions)
        return xy_space, dim_actions, smart_actions

    # ##########################################################################
    # Action Selection
    # ##########################################################################

    def supervised_action(self):
        """
        This method selects a grid point which is the closest to the beacon.
        """

        dx = self._xy_pairs[:,0] - self.beacon_center[0]
        dy = self._xy_pairs[:,1] - self.beacon_center[1]
        distances = np.sqrt(dx**2 + dy**2).round().astype(int)

        # TODO: Check if correct
        closest_pair = np.argmin(distances)
        action_idx = closest_pair
        action = action_idx
        return action, action_idx
