#!/usr/bin/env python3

# gym imports
import gym
import gym_ghost

# python imports
from absl import app
from absl import flags
from absl import logging
import time

# custom imports
from ReplayBuffer import ReplayBuffer
from env_specs import mv2beacon_specs
from agent_specs import compassagent_specs, gridagent_specs
from CompassAgent import CompassAgent
from GridAgent import GridAgent
from squidward import print_squidward
from assets.helperFunctions.timestamps import print_timestamp as print_ts

# pysc2 imports
from pysc2.lib import actions

def main(argv):
    del argv
    print_squidward()
    agent = CompassAgent(compassagent_specs)
    # agent = GridAgent(gridagent_specs)
    env = gym.make("sc2-v0")
    obs, reward, done, info  = env.setup(mv2beacon_specs)

    while(True):
        # Action selection
        action = agent.policy(obs, reward, done, info)

        if (action is 'reset'):
            obs, reward, done, info = env.reset()
        else:
            # Peforming selected action
            obs, reward, done, info = env.step(action)
            agent.evaluate(obs, reward, done, info)

if __name__=="__main__":
    app.run(main)
