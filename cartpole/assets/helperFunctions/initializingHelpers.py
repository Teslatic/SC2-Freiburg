from assets.agents.PendulumAgent import PendulumAgent
from assets.helperFunctions.timestamps import print_timestamp as print_ts


def setup_agent(agent_specs):
    """
    Initializing right agent and agent_interface for the environment.
    """
    agent_type = agent_specs["AGENT_TYPE"]
    if agent_type == 'pendulum':
        return PendulumAgent(agent_specs)
    else:
        print_ts("The agent type {} is not known".format(agent_type))
        exit()
