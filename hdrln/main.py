#!/usr/bin/env python3

# python imports
from absl import app
import time

# gym imports
import gym
import gym_sc2

# custom imports
from specs.agent_specs import agent_specs
from specs.env_specs import env_specs
from assets.helperFunctions.initializingHelpers import setup_agent
from assets.helperFunctions.FileManager import FileManager
from assets.splash.squidward import print_squidward


def main(argv):
    try:
        # FileManager: Save specs and create experiment
        fm = FileManager()
        try:
            fm.create_experiment(agent_specs["EXP_NAME"])  # Automatic cwd switch
            fm.save_specs(agent_specs, env_specs)
        except:
            print("Creating eperiment or saving specs failed.")
            exit()
        fm.create_train_file()

        agent = setup_agent(agent_specs)
        # Loading of model possible fot training mode
        if agent_specs["AGENT_TYPE"] == 'move2beacon':
            env = gym.make("gym-sc2-m2b-v0") # Move2Beacon
        if agent_specs["AGENT_TYPE"] == 'collectmineralshards':
            env = gym.make("gym-sc2-mineralshards-v0")
        if agent_specs["AGENT_TYPE"] == 'defeatroaches':
            env = gym.make("gym-sc2-defeatroaches-v0")

        mode = "learning"
        obs, reward, done, info = env.setup(mv2beacon_specs, mode)

        # Training loop
        while(True):
            action = agent.policy(obs, reward, done, info) # Action selection

            if (action is 'reset'):  # Resetting the environment
                obs, reward, done, info = env.reset()
                if agent.episodes % agent.model_save_period == 0:
                    agent.save_model(fm.get_cwd()) # Cyclic model saving
            else:  # Peforming selected action on environment
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
