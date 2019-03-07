#!/usr/bin/env python3

# python imports
from absl import app
import time

# gym imports
import gym
import gym_sc2

# custom imports
from specs.agent_specs import agent_specs
from specs.env_specs import env_specs # env specs

from assets.helperFunctions.initializingHelpers import setup_agent, setup_multiple_agents, setup_env
from assets.helperFunctions.HDRLNFileManager import HDRLNFileManager
from assets.helperFunctions.timestamps import print_timestamp as print_ts
# ...

# Import architecture files
# architecture_file = ...
# import assets.skills.move2beacon.specs as mv2b_specs
# from assets.skills.collectmineralshards.specs.spec_summary import cMS_specs
#
# from assets.skills.move2beacon.model import mv2b_model
# from assets.skills.collectmineralshards.model import cMS_model

def main(argv):
	try:
		#######################################################################
		# Initializing
		#######################################################################

		# FileManager: Save specs and create experiment #######################
		fm = HDRLNFileManager()
		try:
			fm.create_experiment(agent_specs["EXP_NAME"])  # Auto cwd switch
			fm.save_specs(agent_specs, env_specs)
		except:
			print("Creating experiment or saving specs failed.")
			exit()

		fm.create_train_file()


		# Setup HDRL agent ####################################################
		agent = setup_agent(agent_specs)

		# Extract skills
		# The skill name corresponds to the folder name, where the model and
		# the specs are saved. Skills are saved in the skill directory.
		skill_dir = fm.get_main_dir() + '/skills'
		skill_name_list = ['move2beacon','collectmineralshards']

		# Extract the specs...
		skill_specs_list = fm.extract_skill_specs(skill_name_list)
		# ... and set up the agents...
		agent_list = setup_multiple_agents(skill_specs_list)
		# ... and add skills to the HDRLN agent
		agent.add_skills(skill_dir, skill_name_list, agent_list)
		agent.seal_skill_space()

		# Setup environment ###################################################
		env, obs, reward, done, info  = setup_env(env_specs)

		#######################################################################
		# Learning Loop
		#######################################################################

		while(True):
			# Action selection ################################################
			action = agent.policy(obs, reward, done, info)

			# Performing action ###############################################
			if action is not 'reset':    # Peforming selected action
				obs, reward, done, info = env.step(action)
				dict_agent_report = agent.evaluate(obs, reward, done, info)
				fm.log_training_reports(dict_agent_report)
			else: # Resetting the environment/ End episode
				obs, reward, done, info = env.reset()
				# Save intermediate model
				if agent.episodes % agent.model_save_period == 0:
					agent.save_model(fm.get_cwd())

			# Ending training #################################################
			if env.finished:
				print("Finished learning.")
				break
	# emergency saving when training is manually stopped ######################
	except KeyboardInterrupt:
		agent.save_model(fm.get_cwd(), emergency=True)
		exit()

if __name__ == "__main__":
    # No flags for arg parsing defined yet.
    app.run(main)
