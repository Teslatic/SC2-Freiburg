#!/usr/bin/env python3

# python imports
from absl import app
import time

# gym imports
import gym
import gym_sc2

# custom imports
from specs.agent_specs import agent_specs
from specs.env_specs import mv2beacon_specs # env specs
from assets.helperFunctions.initializingHelpers import setup_agent, setup_multiple_agents
from assets.helperFunctions.FileManager import FileManager
from assets.splash.squidward import print_squidward
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
		# FileManager: Save specs and create experiment
		fm = FileManager()
		try:
			fm.create_experiment(agent_specs["EXP_NAME"])  # Automatic cwd switch
			fm.save_specs(agent_specs, mv2beacon_specs)
		except:
			print("Creating experiment or saving specs failed.")
			exit()
		fm.create_train_file()

		# Create HDRL agent
		agent = setup_agent(agent_specs)

		# Extract skills
		skill_name_list = ['move2beacon','collectmineralshards']

		skill_specs_list = fm.extract_skill_specs(skill_name_list)

		agent_list = setup_multiple_agents(skill_specs_list)
		move2beacon_agent = agent_list[0]
		collectmineralshards_agent = agent_list[1]
		move2beacon_agent.DQN.load(fm.main_dir + '/assets/skills/' + skill_name_list[0] + '/model.pt') # Load model into agent
		collectmineralshards_agent.DQN.load(fm.main_dir + '/assets/skills/' + skill_name_list[1] + '/model.pt') # Load model into agent
		move2beacon_skill = move2beacon_agent.DQN # Extract DQN skill network
		collectmineralshards_skill = collectmineralshards_agent.DQN
		agent.add_skill_list([move2beacon_skill, collectmineralshards_skill])
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
