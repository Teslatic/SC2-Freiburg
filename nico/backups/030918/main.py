#!/usr/bin/env python3


def main(unused_argv):

    # GLOBAL CONSTANTS

    EPISODES = 10000

    BASE_AGENT = 'base_agent'
    SCREEN_DIM = 84
    MINIMAP_DIM = 64
    GAMMA = 1.0
    OPTIM_LR = 0.0001
    BATCH_SIZE = 32
    TARGET_UPDATE_PERIOD = 10
    HIST_LENGTH = 1
    REPLAY_SIZE = 100000
    DEVICE = 'cpu'

    MAP = 'MoveToBeacon'
    PLAYERS = [sc2_env.Agent(sc2_env.Race.terran)]
    STEP_MULTIPLIER = 16  # 16 = 1s game time, None = map default
    GAMESTEPS = None  # 0 = unlimited game time, None = map default
    VISUALIZE = True

    epsilon_file = {
                    'EPSILON': 1.0,
                    'EPS_START': 1.0,
                    'EPS_END': 0.1,
                    'EPS_DECAY': 20000
                    }

    agent_file = {
                  'TYPE': BASE_AGENT,  # Standard: BASE_AGENT
                  'SCREEN_DIM': SCREEN_DIM,  # Standard 84
                  'MINIMAP_DIM': MINIMAP_DIM,  # Standard 64
                  'GAMMA': GAMMA,  # Standard 0.99
                  'OPTIM_LR': OPTIM_LR,  # Standard 0.0001
                  'BATCH_SIZE': BATCH_SIZE,  # Standard 32
                  'TARGET_UPDATE_PERIOD': TARGET_UPDATE_PERIOD,   # Standard 5
                  'HIST_LENGTH': HIST_LENGTH,   # Standard 4
                  'REPLAY_SIZE': REPLAY_SIZE,
                  'DEVICE': DEVICE,
                  'EPSILON_FILE': epsilon_file
                  }

    env_file = {
                'MAP_NAME': MAP,
                'PLAYERS': PLAYERS,
                'STEP_MULTIPLIER': STEP_MULTIPLIER,  # Standard 16
                'GAMESTEPS': GAMESTEPS,
                'VISUALIZE': VISUALIZE
                }

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
        print_ts("Initalizing environment")
        env = sc2_env.SC2Env(
            map_name=env_file['MAP_NAME'],
            players=env_file['PLAYERS'],
            agent_interface_format=agent_interface,
            step_mul=env_file['STEP_MULTIPLIER'],
            game_steps_per_episode=env_file['GAMESTEPS'],
            visualize=env_file['VISUALIZE'])
        return env

    def _xy_locs(mask):
        """
        Mask should be a set of bools from comparison with a feature layer.
        """
        y, x = mask.nonzero()
        return list(zip(x, y))

    try:
        # Initalizing
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        agent_file["DEVICE"] = device
        print_ts("Performing calculations on {}".format(device))

        torch.set_printoptions(linewidth=750, profile="full")

        # Setting up the agent
        agent = BaseAgent(agent_file)
        agent_interface = agent.setup_interface()

        # Setting up the environment
        with setup_env(env_file, agent_interface) as env:
            agent.setup(env.observation_spec(), env.action_spec())  # Necessary? --> For each minigame
            for ep in range(EPISODES):  # Starting an episode
                observation = env.reset()
                actual_obs = observation[0]
                # # FLAGS
                # if ep != 0:  # No flags for the first episode
                #     agent.set_episode_flags(ep)

                # Setting flags and random seed. Clearing reward and history?
                while True:  # Starting a timestep
                    # get beacon center and last_score for debug reference
                    beacon_center = np.mean(_xy_locs(actual_obs.observation.feature_screen.player_relative == 3), axis=0).round()
                    last_score = env._last_score[0]  # Is this not contained in the reward structure?

                    # make one step
                    action = agent.step(actual_obs, beacon_center, last_score)
                    next_obs = env.step(action)
                    actual_obs = next_obs[0]
            # Alle 10 episoden 1-test
            # Neue environment
            # Seed setzen?
            # Neue Schleife Agent step mit greedy actions
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    print("Importing packages")
    # normal python modules
    import numpy as np
    from pysc2.env import sc2_env
    from absl import app
    import torch

    # from os import path
    import sys
    if "../" not in sys.path:
        sys.path.append("../")

    # custom imports
    from assets.agents.BaseAgent import BaseAgent
    from assets.agents.TripleAgent import TripleAgent
    from assets.helperFunctions.timestamps import print_timestamp as print_ts
    print_ts("Starting main")
    app.run(main)
