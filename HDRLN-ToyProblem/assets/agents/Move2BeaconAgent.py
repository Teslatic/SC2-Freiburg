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
    This is a simple agent that uses an PyTorch DQN_module as Q value
    approximator. Current implemented features of the agent:
    - Simple initializing with the help of an agent_specs.
    - Policy switch between imitation and epsilon greedy learning session.
    - Storage of experience into simple Experience Replay Buffer
    - Intermediate saving of model weights
    """
    # ##########################################################################
    # Initializing the agent
    # ##########################################################################

    def __init__(self, agent_specs):
        """
        steps_done: Total timesteps done in the agents lifetime.
        timesteps:  Timesteps performed in the current episode.
        choice:     Choice of epsilon greedy method (None if supervised)
        loss:       Action loss
        hist:       The history buffer for more complex minigames.
        """
        super(Move2BeaconAgent, self).__init__(agent_specs)
     
  

    # ##########################################################################
    # Action Selection
    # ##########################################################################

    def prepare_timestep(self, obs, reward, done, info):
        """
        timesteps:
        """
        super(Move2BeaconAgent, self).prepare_timestep(obs, reward, done, info)

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

