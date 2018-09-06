from pysc2.env import sc2_env
import torch

def setup_agent(agent_file):
    """
    Initializing right agent and agent_interface for the environment.
    """
    agent_type = agent_file["TYPE"]
    if agent_type == 'base_agent':
        print_ts("Using a Base Agent")
        agent = BaseAgent(agent_file)
    if agent_type == 'triple_agent':
        agent = TripleAgent(agent_file)
    else:
        print_ts("The agent type {} is not known".format(agent_type))
        exit()
    return agent, agent.setup_interface()

def setup_env(env_file, agent_interface):
    """
    Refer to PYSC2 documentation.
    """
    env = sc2_env.SC2Env(
        map_name=env_file['MAP_NAME'],
        players=env_file['PLAYERS'],
        agent_interface_format=agent_interface,
        step_mul=env_file['STEP_MULTIPLIER'],
        game_steps_per_episode=env_file['GAMESTEPS'],
        visualize=env_file['VISUALIZE'])
    return env, env_file['EPISODES']

def setup_torch():
    # Initalizing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(linewidth=750, profile="full")
    return device
