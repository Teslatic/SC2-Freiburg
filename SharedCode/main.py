#!/usr/bin/env python3

# python imports
from absl import app
import time

# gym imports
import gym
import gym_ghost

# custom imports
from specs.agent_specs import agent_specs
from specs.env_specs import mv2beacon_specs
from assets.helperFunctions.initializingHelpers import setup_agent
from assets.helperFunctions.FileManager import FileManager
from assets.splash.squidward import print_squidward


def main(argv):
    # print_squidward()

    try:
        # FileManager: Save specs and create experiment
        fm = FileManager()
        try:
            fm.create_experiment(agent_specs["EXP_NAME"])  # Automatic cwd switch
            fm.save_specs(agent_specs, mv2beacon_specs)
        except:
            print("Creating eperiment or saving specs failed.")
            exit()
        fm.create_train_file()

        agent = setup_agent(agent_specs)
        # Loading of model possible fot training mode
        agent.set_supervised_mode()

        # setup environment in learning mode
        env = gym.make("sc2-v0")

        agent.get_env(env)

        obs, reward, done, info = env.setup(mv2beacon_specs, "learning")

        while(True):
            # Action selection
            action = agent.policy(obs, reward, done, info)


            if (action is 'reset'):  # Resetting the environment
                obs, reward, done, info = env.reset()
                if agent.episodes % agent_specs['MODEL_SAVE_PERIOD'] == 0:
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
