from assets.agents.CollectMineralShardsAgent import CollectMineralShardsAgent
from assets.agents.DefeatRoachesAgent import DefeatRoachesAgent
from assets.agents.Move2BeaconAgent import Move2BeaconAgent
from assets.agents.HDRLNAgent import HDRLNAgent

from assets.helperFunctions.timestamps import print_timestamp as print_ts


def setup_agent(agent_specs):
    """
    Initializing right agent and agent_interface for the environment.
    """
    agent_type = agent_specs["AGENT_TYPE"]
    if agent_type == 'defeatroaches':
        return DefeatRoachesAgent(agent_specs)
    if agent_type == 'collectmineralshards':
        return CollectMineralShardsAgent(agent_specs)
    if agent_type == 'move2beacon':
        return Move2BeaconAgent(agent_specs)
    if agent_type == 'hdrl':
        return HDRLNAgent(agent_specs)

    else:
        print_ts("The agent type {} is not known".format(agent_type))
        exit()

def setup_multiple_agents(specs_list):
    """
    Initializes multiple agents according to the specs list
    """
    agent_list = []
    for specs in specs_list:
        agent = setup_agent(specs)
        agent_list.append(agent)
    return agent_list
