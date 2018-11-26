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
from assets.helperFunctions.FileManager import log_test_reports
from assets.helperFunctions.FileManager import save_specs, load_spec_summary
from assets.helperFunctions.FileManager import create_experiment_at_main
from assets.splash.squidward import print_squidward


def main(argv):
    print_squidward()

    # load specs used in experiment
    spec_summary = load_spec_summary(FLAGS.specs)
    agent = setup_agent(spec_summary, mode="testing")

    agent.DQN.load(FLAGS.model)

    env = gym.make("sc2-v0")

    # setup environment in testing mode
    obs, reward, done, info = env.setup(spec_summary, mode="testing")

    while(True):
        # Action selection

        action = agent.policy(obs, reward, done, info)

        if (action is 'reset'):
            obs, reward, done, info = env.reset()
        else:
            # Peforming selected action
            obs, reward, done, info = env.step(action)
            test_report, _ = agent.evaluate(obs, reward, done, info)

            log_test_reports(test_report, spec_summary['ROOT_DIR'])



if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_string("specs",None , "path to spec summary")
    flags.DEFINE_string("model",None , "path to pytorch model")

    app.run(main)
