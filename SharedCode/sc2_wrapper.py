from pysc2.env import sc2_env
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class MoveToBeacon_Wrapper(gym.Wrapaper):
    """
    A wrapper class that uses the PYSC2 environment like a gym environment.
    """
    def __init__(self):
        self.visualize = False
        self.map_name = 'MoveToBeacon'
        self.players = [sc2_env.Agent(sc2_env.Race.terran)]
        self.screen_dim = 84
        self.minimap_dim = 64
        self.step_mul = 4
        self.game_steps = 1920
        self.save_replay = False,
        self.replay_dir = None
        # observation space und action space anlegen
        self.agent_interface = self.setup_interface()
        env = setup_env()

    def step(self, action):
        self.env.step(action)

    def reset(self):
        self.env.reset()

    def setup_interface(self):
        """
        Setting up agent interface for the environment.
        """
        agent_interface = features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=self.screen_dim, minimap=self.minimap_dim),
                            use_feature_units=True)
        return agent_interface

    def setup(env_file, agent_interface):
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
            map_name=self.map_name,
            players=self.players,
            agent_interface_format=self.setup_interface(),
            step_mul=self.step_mul,
            game_steps_per_episode=self.game_steps,
            visualize=self.visualize,
            save_replay_episodes=self.save_replay,
            replay_dir=self.replay_dir
        return env

    def close(self):
        """
        Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
