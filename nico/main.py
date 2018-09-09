#!/usr/bin/env python3


def start_test(test_env_file, net_weights, target_net_weights):
    # print("Starting Test in episode {}".format(ep))
    test_agent = BaseAgent(agent_file)
    test_agent_interface = test_agent.setup_interface()
    test_env = setup_env(test_env_file,test_agent_interface)
    test_agent.setup(test_env.observation_spec(), test_env.action_spec())  # Necessary? --> For each minigame
    # Load parameters into test agent
    test_agent.set_net_weights(net_weights)
    test_agent.set_target_net_weights(target_net_weights)
    # Seed setzen?
    test_observation = test_env.reset()
    actual_test_obs = test_observation[0]
    # Setting flags and random seed. Clearing reward and history?
    while True:  # Starting a timestep
        # get last_score for debug reference
        last_score_test = test_env._last_score[0]  # Is this not contained in the reward structure?
        # make one step
        test_action = test_agent.step(actual_test_obs, last_score_test, 'test')
        next_test_obs = test_env.step(test_action)
        actual_test_obs = next_test_obs[0]
    print("TEST HAS ENDED")


def main(unused_argv):

    # Create a new experiment
    experiment_name = input("Please enter your experiment name: ")
    # experiment_name = 'Multiplot'
    exp_root_dir = create_experiment_at_main(experiment_name)

    # Setting up the torch, agent, agent interface and environment
    print_ts("Performing calculations on {}".format(setup_torch()))
    agent = BaseAgent(agent_file)
    agent_interface = agent.setup_interface()
    env = setup_env(env_file, agent_interface)
    agent.setup(env.observation_spec(), env.action_spec())  # Necessary? --> For each minigame
    observation = env.reset()
    actual_obs = observation[0]
    # try:
    # With statement should be used to properly close the documents and maybe plot graphics in case of an error
    while True:  # Starting a timestep
        # last_score for debug reference
        last_score = env._last_score[0]  # Is this not contained in the reward structure?
        action = [actions.FUNCTIONS.select_army("select")]
        if actual_obs.first():
            # Select Army in first step
            print("First Step")
            action = [actions.FUNCTIONS.select_army("select")]
        if actual_obs.last():
            print("Last step: epsilon is at {}, Total score is at {}".format(agent.epsilon, agent.reward))
            agent.end_episode('learn')
            env.reset()

            # Setting flags and random seed. Clearing reward and history?
            # if d agent flag risen
            #     test_thread = Thread(target=start_test, args=(test_env_file, agent.get_net_weights(), agent.get_target_net_weights()))
            #     test_thread.start()
            # Count new episode. If last episode --> Break
        if not actual_obs.first() and not actual_obs.last():  # make one step
            action = agent.step(actual_obs, last_score, 'learn')

        next_obs = env.step(action)
        actual_obs = next_obs[0]

    # except KeyboardInterrupt:
    #     print_ts("Shutdown by user")
    # except:
    #     print_ts("Problem in the loop")

if __name__ == "__main__":
    # try:
    if True:
        print("Importing packages")
        # normal python modules
        from absl import app
        from threading import Thread
        from os import path
        import sys
        if "../" not in sys.path:
            sys.path.append("../")


        from pysc2.lib import actions
            # custom imports
        from assets.agents.BaseAgent import BaseAgent
        # from assets.agents.TripleAgent import TripleAgent
        from assets.helperFunctions.timestamps import print_timestamp as print_ts
        from assets.helperFunctions.initializingHelpers import setup_torch
        from assets.helperFunctions.FileManager import *
        # from assets.helperFunctions.initializingHelpers import setup_agent
        from assets.helperFunctions.initializingHelpers import setup_env
        from parameterfile import agent_file, env_file, test_env_file
        print_ts("Starting main")
    # except:
    #     print("Some packages might be missing.")

    app.run(main)
