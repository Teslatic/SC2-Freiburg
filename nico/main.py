#!/usr/bin/env python3

def main(unused_argv):

    # Create a new experiment
    experiment_name = input("Please enter your experiment name: ")
    exp_root_dir = create_experiment_at_main(experiment_name)
    agent_file["EXP_PATH"] = exp_root_dir

    # Setting up the agent, agent interface and environment
    agent = CompassAgent(agent_file)
    agent_interface = agent.setup_interface()
    env = setup_env(env_file, agent_interface)
    agent.setup(env.observation_spec(), env.action_spec())  # Necessary? --> For each minigame

    try:
        # Reset environment and readout initial observation
        observation = env.reset()
        actual_obs = observation[0]  # Only most recent observation
        start_time = time.time()
        while True:  # Perform a timestep

            action = agent.policy(actual_obs)

            if (action is 'reset'):
                next_obs = env.reset()
                end_time = time.time()
                print_ts("Episode took {} seconds.".format(end_time-start_time))
                start_time = time.time()
                # raise Exception("Test for weight saving")
            else:
                # Peforming selected action
                next_obs = env.step(action)

            if not actual_obs.first() and not actual_obs.last():  # make one step
                # Saving the episode data. Pushing the information onto the memory.
                agent.store_transition(next_obs[0])

                # Optimize the agent
                if agent.get_memory_length() >= 100:
                    agent.optimize()

                    # Print actual status information
                    if not agent.silentmode:
                        agent.print_status()

            if agent.logging:
                agent.log()

            actual_obs = next_obs[0]
            # end_time = time.time()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, saving model and data!")
        agent._save_model(emergency=True)
        print("Saving was successful")
    except:
        print("Fault occured. Emergency saving of agent weights")
        agent._save_model(emergency=True)
        print("Saving was successful")

if __name__ == "__main__":
    try:
    # if True:
        print("Importing packages")
        # normal python modules
        from absl import app
        print("no problems till os")
        from os import path
        import sys
        import time
        if "../" not in sys.path:
            sys.path.append("../")
        print("no problems till pysc2")
        from pysc2.lib import actions

        # custom imports
        from assets.agents.BaseAgent import CompassAgent
        from assets.helperFunctions.timestamps import print_timestamp as print_ts
        from assets.helperFunctions.FileManager import *
        from assets.helperFunctions.initializingHelpers import setup_env
        from parameterfile import agent_file, env_file, test_env_file
        print_ts("Starting main")
    except:
        print_ts("Fault while loading the modules. Some packages might be missing.")

    app.run(main)

# 
# def start_test(test_env_file, net_weights, target_net_weights):
#     """
#     Used to start an intermediate test with only exploiting actions.
#     """
#     # print("Starting Test in episode {}".format(ep))
#     test_agent = CompassAgent(agent_file)
#     test_agent_interface = test_agent.setup_interface()
#     test_env = setup_env(test_env_file,test_agent_interface)
#     test_agent.setup(test_env.observation_spec(), test_env.action_spec())  # Necessary? --> For each minigame
#     # Load parameters into test agent
#     test_agent.set_net_weights(net_weights)
#     test_agent.set_target_net_weights(target_net_weights)
#     # Seed setzen?
#     test_observation = test_env.reset()
#     actual_test_obs = test_observation[0]
#     # Setting flags and random seed. Clearing reward and history?
#     while True:  # Starting a timestep
#         # get last_score for debug reference
#         # make one step
#         test_action = test_agent.policy(actual_test_obs, last_score_test, 'test')
#         next_test_obs = test_env.step(test_action)
#         actual_test_obs = next_test_obs[0]
#     print("TEST HAS ENDED")
