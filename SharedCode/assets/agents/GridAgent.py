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
    def __init__(self, agent_file, mode='learning'):
        """
        Refer to Move2BeaconAgent.
        """
        super(GridAgent, self).__init__(agent_file, mode)

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
        x_space = np.linspace(0, 83, self.grid_factor, dtype=int)
        y_space = np.linspace(0, 63, self.grid_factor, dtype=int)
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
        self.x_coord = self.beacon_center[0]
        self.y_coord = self.beacon_center[1]

        distances = []
        for xy_pair in self._xy_pairs:
            dx = np.abs(xy_pair[0] - self.beacon_center[0])
            dy = np.abs(xy_pair[1] - self.beacon_center[1])
            distances.append(np.sqrt(dx**2 + dy**2).round())

        # TODO: Check if correct
        closest_pair = np.argmin(distances)
        self.action_idx = closest_pair
        return self.action_idx, self.action_idx

    def log(self):
        pass
        buffer_size = 10  # This makes it so changes appear without buffering
        with open('output.log', 'w', buffer_size) as f:
                f.write('{}\n'.format(self.feature_screen))

    def _save_model(self, emergency=False):
        if emergency:
            save_path = self.exp_path + "/model/emergency_model.pt"
        else:
            save_path = self.exp_path + "/model/model.pt"

        # Path((self.exp_path + "/model")).mkdir(parents=True, exist_ok=True)
        self.DQN.save(save_path)
