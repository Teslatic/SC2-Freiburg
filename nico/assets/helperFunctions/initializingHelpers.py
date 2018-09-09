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
    From the official PYSC2 documentation:
        You must pass a resolution that you want to play at. You can send either
    feature layer resolution or rgb resolution or both. If you send both you
    must also choose which to use as your action space. Regardless of which you
    choose you must send both the screen and minimap resolutions.

    For each of the 4 resolutions, either specify size or both width and
    height. If you specify size then both width and height will take that value.

    Args:
      _only_use_kwargs: Don't pass args, only kwargs.
      map_name: Name of a SC2 map. Run bin/map_list to get the full list of
          known maps. Alternatively, pass a Map instance. Take a look at the
          docs in maps/README.md for more information on available maps.
      players: A list of Agent and Bot instances that specify who will play.
      agent_race: Deprecated. Use players instead.
      bot_race: Deprecated. Use players instead.
      difficulty: Deprecated. Use players instead.
      screen_size_px: Deprecated. Use agent_interface_formats instead.
      minimap_size_px: Deprecated. Use agent_interface_formats instead.
      agent_interface_format: A sequence containing one AgentInterfaceFormat
        per agent, matching the order of agents specified in the players list.
        Or a single AgentInterfaceFormat to be used for all agents.
      visualize: Whether to pop up a window showing the camera and feature
          layers. This won't work without access to a window manager.
      step_mul: How many game steps per agent step (action/observation). None
          means use the map default.
      save_replay_episodes: Save a replay after this many episodes. Default of 0
          means don't save replays.
      replay_dir: Directory to save replays. Required with save_replay_episodes.
      game_steps_per_episode: Game steps per episode, independent of the
          step_mul. 0 means no limit. None means use the map default.
      random_seed: Random number seed to use when initializing the game. This
          lets you run repeatable games/tests.
    """
    env = sc2_env.SC2Env(
        map_name=env_file['MAP_NAME'],
        players=env_file['PLAYERS'],
        agent_interface_format=agent_interface,
        step_mul=env_file['STEP_MULTIPLIER'],
        game_steps_per_episode=env_file['GAMESTEPS'],
        visualize=env_file['VISUALIZE'],
        save_replay_episodes=env_file['SAVE_REPLAY'],
        replay_dir=env_file['REPLAY_DIR'])
    return env

def setup_torch():
    """
    Setting GPU if available. Else, use the CPU.
    """
    # Initalizing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(linewidth=750, profile="full")
    return device
