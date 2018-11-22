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
from assets.helperFunctions.FileManager import log_reports, load_spec_summary
from assets.splash.squidward import print_squidward


def main(argv):
    print_squidward()
    spec_summary = load_spec_summary(FLAGS.specs)
    print(type(spec_summary['GRID_FACTOR']))
    exit()
    agent = setup_agent(spec_summary)
    env = gym.make("sc2-v0")
    obs, reward, done, info = env.setup(spec_summary)

    while(True):
        # Action selection
        action = agent.policy(obs, reward, done, info)

        if (action is 'reset'):
            obs, reward, done, info = env.reset()
        else:
            # Peforming selected action
            obs, reward, done, info = env.step(action)
            dict_agent_report, exp_root_dir = agent.evaluate(obs, reward, done, info)

            log_reports(dict_agent_report, exp_root_dir)



if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_string("specs",None , "path to spec summary")
    app.run(main)
