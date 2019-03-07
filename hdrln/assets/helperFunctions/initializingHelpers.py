from assets.agents.CollectMineralShardsAgent import CollectMineralShardsAgent
from assets.agents.DefeatRoachesAgent import DefeatRoachesAgent
from assets.agents.Move2BeaconAgent import Move2BeaconAgent
from assets.hdrln.HDRLNAgent import HDRLNAgent

from assets.helperFunctions.timestamps import print_timestamp as print_ts

# gym imports
import gym
import gym_sc2

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

def setup_env(env_specs):
    """
    Initializing right agent and agent_interface for the environment.
    """
    env_id = env_specs["ENV_ID"]
    env = gym.make(env_id)
    obs, reward, done, info = env.setup(env_specs)
    return env, obs, reward, done, info


def setup_multiple_envs(specs_list):
    """
    Initializes multiple agents according to the specs list
    """
    env_list = []
    for specs in specs_list:
        env = setup_env(specs)
        env_list.append(env)
    return env_list
