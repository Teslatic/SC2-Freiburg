from assets.agents.CollectMineralShardsAgent import CollectMineralShardsAgent
from assets.agents.DefeatRoachesAgent import DefeatRoachesAgent
from assets.agents.Move2BeaconAgent import Move2BeaconAgent
from assets.hdrln.HDRLNAgent import HDRLNAgent
from assets.helperFunctions.FileManager import FileManager
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

def setup_env(env_specs, mode='learning'):
    """
    Initializing right agent and agent_interface for the environment.
    """
    env_id = str(env_specs["ENV_ID"])
    if  env_id == 'move2beacon':
        env = gym.make("gym-sc2-m2b-v0") # Move2Beacon
    elif env_id == 'collectmineralshards':
        env = gym.make("gym-sc2-mineralshards-v0")
    elif env_id == 'defeatroaches':
        env = gym.make("gym-sc2-defeatroaches-v0")
    else:
        print_ts("Define a valid environment id in the agent specs!")
        exit()
    # mode = env_specs["MODE"]
    obs, reward, done, info = env.setup(env_specs, mode)
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

def setup_fm(agent_specs, env_specs):
    """
    Initializes the FileManager which is responsible for storing data.
    """
    fm = FileManager()
    try:
        fm.create_experiment(agent_specs["EXP_NAME"])  # Automatic cwd switch
        fm.save_specs(agent_specs, env_specs)
    except:
        print("Creating eperiment or saving specs failed.")
        exit()
    fm.create_train_file()
    return fm
