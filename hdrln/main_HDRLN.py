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
from assets.helperFunctions.initializingHelpers import setup_agent, setup_multiple_agents
from assets.helperFunctions.FileManager import FileManager
from assets.splash.squidward import print_squidward
from assets.helperFunctions.timestamps import print_timestamp as print_ts
# ...

# Import architecture files
# architecture_file = ...
# import assets.skills.move2beacon.specs as mv2b_specs
# from assets.skills.collectmineralshards.specs.spec_summary import cMS_specs
#
# from assets.skills.move2beacon.model import mv2b_model
# from assets.skills.collectmineralshards.model import cMS_model

def load_models(skill_dir, skill_name_list, agent_list):
	"""
	Loads the skills (models) at the skill directory into the agents in the agent list.
	"""
	print(skill_name_list)
	N_agent = len(agent_list)
	for idx in range(N_agent):
		skill_name = skill_name_list[idx]
		# Load model into agent
		agent_list[idx].load_model(skill_dir + '/' + skill_name + '/model.pt')

def extract_skills(agent_list):
	"""
	Loads the skills (models) at the skill directory into the agents in the agent list.
	"""
	skill_list =[]
	for idx in range(len(agent_list)):
		skill_list.append(agent_list[idx].extract_skill()) # Extract DQN skill network
	return skill_list

def main(argv):
	try:
		# FileManager: Save specs and create experiment
		fm = FileManager()
		try:
			fm.create_experiment(agent_specs["EXP_NAME"])  # Automatic cwd switch
			fm.save_specs(agent_specs, env_specs)
		except:
			print("Creating experiment or saving specs failed.")
			exit()
		fm.create_train_file()

		# Create HDRL agent
		agent = setup_agent(agent_specs)

		# Extract skills
		# The skill name corresponds to the folder name, where the model and the specs are saved. Skills are saved in the skill directory.
		skill_name_list = ['move2beacon','collectmineralshards']

		# Extract the specs...
		skill_specs_list = fm.extract_skill_specs(skill_name_list)
		# ... and set up the agents
		agent_list = setup_multiple_agents(skill_specs_list)

		skill_dir = fm.get_main_dir() + '/assets/skills'
		load_models(skill_dir, skill_name_list, agent_list)

		skill_list = extract_skills(agent_list)
		agent.add_skill_list(skill_list)
		print_ts("Code ran through")
		exit()

		# Create environment
		env = gym.make("gym-sc2-m2b-v0")
		obs, reward, done, info = env.setup(mv2beacon_specs, "learning")

		while(True):
        	# Action selection
			action = agent.policy(obs, reward, done, info)


			if (action is 'reset'):  # Resetting the environment
				obs, reward, done, info = env.reset()
				if agent.episodes % agent.model_save_period == 0:
					agent.save_model(fm.get_cwd())
			else:  # Peforming selected action
				obs, reward, done, info = env.step(action)
				dict_agent_report = agent.evaluate(obs, reward, done, info)
				fm.log_training_reports(dict_agent_report)

			if env.finished:
				print("Finished learning.")
				break
	except KeyboardInterrupt:
		agent.save_model(fm.get_cwd(), emergency=True)
		exit()

if __name__ == "__main__":
    # No flags for arg parsing defined yet.
    app.run(main)
