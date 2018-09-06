#!/usr/bin/env python3

def start_test(agent, test_env_file, agent_interface, ep):
        print("Starting Test in episode {}".format(ep))
        test_env, test_episodes = setup_env(test_env_file, agent_interface)
        # Seed setzen?
        for test_ep in range(test_episodes):  # Starting an episode
            test_observation = test_env.reset()
            actual_test_obs = test_observation[0]
            # Setting flags and random seed. Clearing reward and history?
            for steps in range(1920):  # Starting a timestep
                # get beacon center and last_score for debug reference
                last_score_test = test_env._last_score[0]  # Is this not contained in the reward structure?
                # make one step
                test_action = agent.step(actual_test_obs, last_score_test, 'test')
                next_test_obs = test_env.step(test_action)
                actual_test_obs = next_test_obs[0]
        print("TEST HAS ENDED")

def main(unused_argv):
    # try:
    # Setting up the torch, agent, agent interface and environment
    print_ts("Performing calculations on {}".format(setup_torch()))
    agent = BaseAgent(agent_file)
    agent_interface = agent.setup_interface()
    # except:
        # print_ts("Problem while initializing agent or environment.")
        # exit()
    if True:
        # With statement should be used to properly close the documents and maybe plot graphics in case of an error
        env, episodes = setup_env(env_file, agent_interface)
        agent.setup(env.observation_spec(), env.action_spec())  # Necessary? --> For each minigame
        for ep in range(episodes):  # Starting an episode
            print_ts("Starting episode {}".format(ep))
            observation = env.reset()
            actual_obs = observation[0]
            # Setting flags and random seed. Clearing reward and history?
            while True:  # Starting a timestep
                # last_score for debug reference
                last_score = env._last_score[0]  # Is this not contained in the reward structure?

                # make one step
                action = agent.step(actual_obs, last_score, 'learn')
                next_obs = env.step(action)
                actual_obs = next_obs[0]
                #
                if actual_obs.step_type == 2 and ep % 2 == 0:
                    test_thread = Thread(target=start_test, args=(agent, test_env_file, agent_interface, ep))
                    test_thread.start()
    # except KeyboardInterrupt:
    #     print_ts("Shutdown by user")
    # except:
    #     print_ts("Problem in the loop")

if __name__ == "__main__":
    # try:
    if True:
        print("Importing packages")
        # normal python modules
        import numpy as np
        from pysc2.env import sc2_env
        from absl import app
        from threading import Thread
        import sys
        if "../" not in sys.path:
            sys.path.append("../")

            # custom imports
        from assets.agents.BaseAgent import BaseAgent
        from assets.agents.TripleAgent import TripleAgent
        from assets.helperFunctions.timestamps import print_timestamp as print_ts
        from assets.helperFunctions.initializingHelpers import setup_torch
        from assets.helperFunctions.initializingHelpers import setup_agent
        from assets.helperFunctions.initializingHelpers import setup_env
        from parameterfile import epsilon_file, agent_file, env_file, test_env_file
        print_ts("Starting main")
    # except:
    #     print("Some packages might be missing.")

    app.run(main)
