#!/usr/bin/env python3

import gym
import gym_ghost
from absl import app
from absl import flags
from absl import logging
import time

from pysc2.lib import actions

def main(argv):
    del argv
    env = gym.make("sc2-v0")

    obs, reward, done, info  = env.reset()

    while(True):
        obs, reward, done, info = env.step(10)
        time.sleep(0.1)

if __name__=="__main__":
    app.run(main)


