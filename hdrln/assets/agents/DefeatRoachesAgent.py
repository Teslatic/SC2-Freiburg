# python imports
from assets.agents.DQNBaseAgent import DQNBaseAgent

# custom imports
from assets.helperFunctions.timestamps import print_timestamp as print_ts


class DefeatRoachesAgent(DQNBaseAgent):
    """
    This agent is specialized on solving the DefeatRoaches task from the
    PYSC2 framework.
    """
    # ##########################################################################
    # Initializing the agent
    # ##########################################################################
    def __init__(self, agent_specs):
        """
        Refer to Move2BeaconAgent.
        """
        super(DefeatRoachesAgent, self).__init__(agent_specs)


    def setup_dqn(self, module_specs):
        """
        Using the constructor of the base class.
        """
        print_ts("Setup DQN of DefeatRoachesAgent")
        super(DefeatRoachesAgent, self).setup_dqn(module_specs)

    def supervised_action(self):
        """
        This method selects an action which will determine the best spot to
        attack.
        Further improvements are possible.
        """
        raise NotImplementedError
