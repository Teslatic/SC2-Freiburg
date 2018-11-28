#!/usr/bin/env python3

# python imports
from absl import app, flags

# gym imports
import gym
import gym_ghost

# custom imports
from specs.agent_specs import agent_specs
from specs.env_specs import mv2beacon_specs
from assets.helperFunctions.initializingHelpers import setup_agent

from assets.helperFunctions.FileManager import FileManager
# from assets.helperFunctions.FileManager import log_test_reports
# from assets.helperFunctions.FileManager import save_specs, load_spec_summary
# from assets.helperFunctions.FileManager import create_experiment_at_main
from assets.splash.squidward import print_squidward


def main(argv):
    print_squidward()

    # load specs used in experiment

    # FileManager
    fm = FileManager()
    spec_summary = fm.load_spec_summary(FLAGS.specs)
    fm.change_cwd(spec_summary["ROOT_DIR"])

    agent = setup_agent(spec_summary)

    agent.DQN.load(FLAGS.model)
    agent.set_testing_mode()
    # agent.specify_experiment_path(spec_summary)

    # setup environment in testing mode
    env = gym.make("sc2-v0")
    obs, reward, done, info = env.setup(spec_summary, "testing")

    fm.create_test_file()
    while(True):
        # Action selection

        action = agent.policy(obs, reward, done, info)

        if (action is 'reset'):
            obs, reward, done, info = env.reset()
        else:
            # Peforming selected action
            obs, reward, done, info = env.step(action)
            test_report = agent.evaluate(obs, reward, done, info)

            fm.log_test_reports(test_report)



if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_string("specs",None , "path to spec summary")
    flags.DEFINE_string("model",None , "path to pytorch model")

    app.run(main)
