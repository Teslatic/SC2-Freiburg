# python imports
import numpy as np
from assets.agents.DQNBaseAgent import DQNBaseAgent

# custom imports
from assets.helperFunctions.timestamps import print_timestamp as print_ts

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)


class CollectMineralShardsAgent(DQNBaseAgent):
    """
    This agent is specialized on solving the CollectMineralShard task from the
    PYSC2 framework.
    """
    # ##########################################################################
    # Initializing the agent
    # ##########################################################################

    def __init__(self, agent_specs):
        """
        Using the constructor of the base class.
        """
        super(CollectMineralShardsAgent, self).__init__(agent_specs)

    def setup_dqn(self):
        print_ts("Setup DQN of CollectMineralShardsAgent")
        super(CollectMineralShardsAgent, self).setup_dqn()

    def supervised_action(self):
        """
        This method selects an action which will force the marine in the
        direction of the beacon.
        Further improvements are possible.
        """
        raise NotImplementedError
