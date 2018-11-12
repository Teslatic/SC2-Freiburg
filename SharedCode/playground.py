#!/usr/bin/env python3

import gym
from absl import app
from absl import flags
from absl import logging
import time

def main(argv):
    del argv
    env = gym.make("sc2-v0")

    observation = env.reset()
    actual_obs = observation[0]
    print(actual_obs.observation.feature_screen.player_relative)
    exit()

    while(True):
        env.step()


if __name__=="__main__":
    app.run(main)


