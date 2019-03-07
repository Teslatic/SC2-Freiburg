# python imports
import numpy as np
from sys import getsizeof
import time

# custom imports
from assets.agents.DQNBaseAgent import DQNBaseAgent
from assets.RL.DQN_module import DQN_module
from assets.helperFunctions.timestamps import print_timestamp as print_ts
# from assets.helperFunctions.FileManager import create_experiment_at_main

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)


class Move2BeaconAgent(DQNBaseAgent):
    """
    This agent is specialized on solving the Move2Beacon task from the
    PYSC2 framework.
    """
    # ##########################################################################
    # Initializing the agent
    # ##########################################################################

    def __init__(self, agent_specs):
        """
        Using the constructor of the base class.
        """
        super(Move2BeaconAgent, self).__init__(agent_specs)

    def setup_dqn(self):
        print_ts("Setup DQN of Move2BeaconAgent")
        super(Move2BeaconAgent, self).setup_dqn()


    # ##########################################################################
    # Action Selection
    # ##########################################################################

    def prepare_timestep(self, obs, reward, done, info):
        """
        timesteps:
        """
        super(Move2BeaconAgent, self).prepare_timestep(obs, reward, done, info)
        # obs for supervised learning
        self.distance = obs[6]
        self.marine_center = obs[7]
        self.beacon_center = obs[8]


    def supervised_action(self):
        """
        This method selects an action which will force the marine in the
        direction of the beacon.
        Further improvements are possible.
        """
        raise NotImplementedError
