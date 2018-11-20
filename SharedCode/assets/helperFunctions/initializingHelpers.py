from GridAgent import GridAgent
from CompassAgent import CompassAgent
from assets.helperFunctions.print_timestamp import print_timestamps as print_ts


def setup_agent(agent_file):
    """
    Initializing right agent and agent_interface for the environment.
    """
    agent_type = agent_file["AGENT_TYPE"]
    if agent_type == 'compass':
        return CompassAgent(agent_file)
    if agent_type == 'grid':
        return GridAgent(agent_file)
    if agent_type == 'original':
        raise("This agent type has not been implemeted yet.")
        exit()
    else:
        print_ts("The agent type {} is not known".format(agent_type))
        exit()
