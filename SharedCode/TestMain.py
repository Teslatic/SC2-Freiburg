#!/usr/bin/env python3

# python imports
from absl import app, flags

# gym imports
import gym
import gym_ghost

# custom imports
# from specs.agent_specs import agent_specs
# from specs.env_specs import mv2beacon_specs
from assets.helperFunctions.initializingHelpers import setup_agent
from assets.helperFunctions.FileManager import FileManager
from assets.splash.squidward import print_squidward


def main(argv):
    # print_squidward()

    # FileManager: load specs used in experiment
    fm = FileManager()
    try:
        spec_summary = fm.load_spec_summary(FLAGS.specs)
        print(spec_summary)
        fm.change_cwd(spec_summary["ROOT_DIR"])
    except:
        print("Loading specs/model failed. Have you selected the right path?")
        exit()
    fm.create_test_file()

    agent = setup_agent(spec_summary)
    agent.DQN.load(FLAGS.model)
    agent.set_testing_mode()


    # setup environment in testing mode
    env = gym.make("sc2-v0")
    obs, reward, done, info = env.setup(spec_summary, "testing")

    while(True):
        # Action selection
        action = agent.policy(obs, reward, done, info)
        print(action)

        if (action is 'reset'):  # Resetting the environment
            obs, reward, done, info = env.reset()
            # No saving of model in test_mode
        else:  # Peforming selected action
            obs, reward, done, info = env.step(action)
            test_report = agent.evaluate(obs, reward, done, info)
            fm.log_test_reports(test_report)

        if env.finished:
            print("Finished testing.")
            break


if __name__ == "__main__":
    # Arg parsing for model and specs paths
    FLAGS = flags.FLAGS
    flags.DEFINE_string("specs", None, "path to spec summary")
    flags.DEFINE_string("model", None, "path to pytorch model")

    app.run(main)
