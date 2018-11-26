# python imports
import numpy as np

# custom imports
from assets.RL.DQN_module import DQN_module
from assets.agents.smart_actions import SMART_ACTIONS_COMPASS as SMART_ACTIONS
from assets.helperFunctions.timestamps import print_timestamp as print_ts
from assets.agents.Move2BeaconAgent import Move2BeaconAgent


class CompassAgent(Move2BeaconAgent):
    """
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
        super(CompassAgent, self).__init__(agent_specs)

    def setup_dqn(self):
        """
        Setting up the DQN module with the compass action space.
        """
        self.smart_actions = SMART_ACTIONS
        self.dim_actions = len(self.smart_actions)
        self.DQN = DQN_module(self.batch_size,
                              self.gamma,
                              self.history_length,
                              self.size_replaybuffer,
                              self.optim_learning_rate,
                              self.dim_actions)
        self.device = self.DQN.device
        print_ts("DQN module has been initalized")

    # ##########################################################################
    # Action Selection
    # ##########################################################################

    def supervised_action(self):
        """
        This method selects an action which will force the marine in the
        direction of the beacon.
        Further improvements are possible.
        """
        relative_vector = self.marine_center - self.beacon_center
        action_choice = np.random.choice([True, False], p=[0.5, 0.5])

        right_to_beacon = relative_vector[0] > 0.
        left_to_beacon = relative_vector[0] < 0.
        vertically_aligned = relative_vector[0] == 0.

        below_beacon = relative_vector[1] > 0.
        above_beacon = relative_vector[1] < 0.
        horizontally_aligned = relative_vector[1] == 0.

        left = 0
        up = 1
        right = 2
        down = 3

        if (horizontally_aligned and vertically_aligned):
            # Hitting beacon, stay on last action
            # self.action_idx not defined in first timestep
            last_action_idx = self.action_idx
            action_idx = last_action_idx
        if (horizontally_aligned and right_to_beacon):  # East
            action_idx = left
        if (below_beacon and vertically_aligned):  # South
            action_idx = up
        if (horizontally_aligned and left_to_beacon):  # West
            action_idx = right
        if (above_beacon and vertically_aligned):  # North
            action_idx = down
        if above_beacon and left_to_beacon:  # North-West
            if action_choice:
                action_idx = right
            if not action_choice:
                action_idx = down
        if (above_beacon and right_to_beacon):  # North-East
            if action_choice:
                action_idx = left
            if not action_choice:
                action_idx = down
        if (below_beacon and right_to_beacon):  # South-East
            if action_choice:
                action_idx = left
            if not action_choice:
                action_idx = up
        if (below_beacon and left_to_beacon):  # South-West
            if action_choice:
                action_idx = right
            if not action_choice:
                action_idx = up
        chosen_action = SMART_ACTIONS[action_idx]
        return chosen_action, action_idx

    # ##########################################################################
    # Print status information of timestep
    # ##########################################################################

    def _save_model(self, emergency=False):
        if emergency:
            save_path = self.exp_path + "/model/emergency_model.pt"
        else:
            save_path = self.exp_path + "/model/model.pt"

        # Path((self.exp_path + "/model")).mkdir(parents=True, exist_ok=True)
        self.DQN.save(save_path)
