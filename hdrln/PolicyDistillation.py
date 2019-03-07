#!/usr/bin/env python3

# python imports
from absl import app
import time

# custom imports
from specs.agent_specs import agent_specs
from specs.env_specs import env_specs # env specs

from assets.helperFunctions.initializingHelpers import setup_agent
from assets.helperFunctions.initializingHelpers import setup_multiple_agents
from assets.helperFunctions.initializingHelpers import setup_env
from assets.helperFunctions.initializingHelpers import setup_multiple_envs

from assets.helperFunctions.HDRLNFileManager import HDRLNFileManager
from assets.helperFunctions.timestamps import print_timestamp as print_ts
# ...

# Import architecture files
# architecture_file = ...
#
# from assets.skills.move2beacon.model import mv2b_model
# from assets.skills.collectmineralshards.model import cMS_model

def main(argv):
    try:
        fm = HDRLNFileManager() # FileManager: Save specs and create experiment
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
        # The skill name corresponds to the folder name, where the model and
        # the specs are saved. Skills are saved in the skill directory.
        skill_dir = fm.get_main_dir() + '/skills'
        skill_name_list = ['move2beacon','collectmineralshards']

        # Extract the specs...
        skill_specs_list = fm.extract_skill_specs(skill_name_list)
        # ... and set up the agents...
        agent_list = setup_multiple_agents(skill_specs_list)
        # ... and add skills to the HDRLN agent
        agent.add(skill_dir, skill_name_list, agent_list)

        # Hand-coded
        mv2b_specs = fm.load_spec_summary(skill_dir + "/move2beacon/specs.csv")
        cMS_specs = fm.load_spec_summary(skill_dir + "/collectmineralshards/specs.csv")

        env_spec_list = [mv2b_specs, cMS_specs]
        # Create all environment
        env_list = setup_multiple_envs(env_spec_list)

        loops = agent.N_skills * 10
        while(True):
            # This is the round robin teacher loop.
            for i in range(loops):
                active_idx = i%agent.N_skills
                active_env = env_list[active_idx]
                agent.lock_active_skill(active_idx)

                # Action selection
                action = agent.policy(obs, reward, done, info, policy_dist=True)
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
