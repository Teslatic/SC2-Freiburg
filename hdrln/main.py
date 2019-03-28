#!/usr/bin/env python3

# python imports
from absl import app

# gym imports
import gym
import gym_sc2

# custom imports
from specs.agent_specs import agent_specs
from specs.env_specs import env_specs
from assets.helperFunctions.initializingHelpers import setup_agent, setup_env, setup_fm
from assets.splash.squidward import print_squidward


def main(argv):
    try:
		#######################################################################
		# Initializing
		#######################################################################

		# FileManager: Save specs and create experiment #######################
        fm = setup_fm(agent_specs, env_specs)

		# Agent: Initializing the agent with its specs ########################
        agent = setup_agent(agent_specs)
        # TODO: Loading of model possible for training mode

		# Setup environment ###################################################
        env, obs, reward, done, info = setup_env(env_specs)

		#######################################################################
		# Learning Loop
		#######################################################################

        while(True):
            action = agent.policy(obs, reward, done, info) # Action selection

            if not (action is 'reset'):  # Peforming action on environment
                obs, reward, done, info = env.step(action)
                dict_agent_report = agent.evaluate(obs, reward, done, info)
                fm.log_training_reports(dict_agent_report)

            else:
                obs, reward, done, info = env.reset() # Reset the environment
                if agent.episodes % agent.model_save_period == 0:
                    agent.save_model(fm.get_cwd()) # Cyclic model saving

            if env.finished: # Ending Training
                print("Finished learning.")
                break
    except KeyboardInterrupt:
        agent.save_model(fm.get_cwd(), emergency=True)
        exit()

if __name__ == "__main__":
    # No flags for arg parsing defined yet.
    app.run(main)
