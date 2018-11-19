#!/usr/bin/env python3
"""
Nico Ott, Hendrik Vloet
Copyright (C) 2018 Nico Ott, Hendrik Vloet
Public Domain
"""
# ______________________________________________________________________________

## \mainpage StarCraft 2 Reinforcement Learning Project
#  Albert Ludwigs Universität Freiburg
#  Project at the chair of Neurorobotics
#  Supervisor: Prof. Dr. J. Bödecker
#  Students: Nico Ott, Hendrik Vloet

print("Importing python modules")
try:
    # Import standard python modules
    from os import path
    import sys
    from absl import app, flags, logging
    import time

    # Import custom python modules
    from pysc2.lib import actions
    import gym
    import gym_sc2
        # custom imports
    if "../" not in sys.path:
        sys.path.append("../")
    from assets.agents.BaseAgent import CompassAgent
    from assets.helperFunctions.timestamps import print_timestamp as print_ts
    from assets.helperFunctions.FileManager import *
    from assets.helperFunctions.initializingHelpers import setup_env
    from parameterfile import agent_file, env_file, test_env_file
    print_ts("Modules loaded. Starting main")
except:
    print_ts("Fault while loading the modules. Some packages might be missing.")


def main(unused_argv):
    del unused_argv

    # Create a new experiment
    experiment_name = input("Please enter your experiment name: ")
    exp_root_dir = create_experiment_at_main(experiment_name)
    agent_file["EXP_PATH"] = exp_root_dir

    # Setting up the agent, agent interface and environment
    agent = CompassAgent(agent_file)
    # agent_interface = agent.setup_interface()
    env = gym.make("sc2-v0")
    ## agent.setup(env.observation_spec(), env.action_spec())  # Necessary? --> For each minigame

    try:
        # Reset environment and readout initial observation
        observation = env.reset()
        actual_obs = observation[0]  # Only most recent observation
        start_time = time.time()
        while True:  # Perform a timestep

            action = agent.policy(actual_obs)


            if (action is 'reset'):
                print_ts("About to reset the environment")
                next_obs = env.reset()
                print_ts("Environment reset. Episode finished.")
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
    # except:
    #     print("Fault occured. Emergency saving of agent weights")
    #     agent._save_model(emergency=True)
    #     print("Saving was successful")

    observation = env.reset()
    actual_obs = observation[0]
    print(actual_obs.observation.feature_screen.player_relative)
    exit()

    while(True):
        env.step()


if __name__=="__main__":
    app.run(main)
