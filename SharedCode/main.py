#!/usr/bin/env python3
"""
Nico Ott, Hendrik Vloet
Copyright (C) 2018 Nico Ott, Hendrik Vloet
Public Domain
"""
# ______________________________________________________________________________

## \mainpage StarCraft 2 Reinforcement Learning Project
#  Albert Ludwigs Universität Freiburg  
#  Project at the chair of Neurorobotics  
#  Supervisor: Prof. Dr. J. Bödecker  
#  Students: Nico Ott, Hendrik Vloet  

# custom imports
# from tensorforce.contrib.openai_gym import OpenAIGym
# import sc2gym
import gym
import sc2gym.envs

# normal python imports
from absl import app
from absl import flags
from absl import logging
import random
import numpy as np
import math
from itertools import count
from collections import namedtuple
import os
from pathlib import Path
import csv

# torch imports
import torch

class MoveToBeacon1d(BaseExample):
    def __init__(self, visualize=False, step_mul=None) -> None:
        super().__init__(_ENV_NAME, visualize, step_mul)

    def get_action(self, env, obs):
        neutral_y, neutral_x = (obs[0] == _PLAYER_NEUTRAL).nonzero()
        if not neutral_y.any():
            raise Exception('Beacon not found!')
        target = [int(neutral_x.mean()), int(neutral_y.mean())]
        target = np.ravel_multi_index(target, obs.shape[1:])
        return target





## @package main
#  Documentation for the main file.
#
#  The main file serves only to read in hyperparemeters, construct architecture & agent objects and finally run the agent with the agent.play() method

## Reads in several flags to pass it to the running app (pysc2).
# Flag Description
# - learning_rate, float: learning for the optimizer
# - gamma, float: Discount factor
# - batch_size, unsigned int: Batch size of the samples durch experience replay
# - target_update, unsigned int: Update Target network every N episodes
# - epochs, unsigned int: amount of episodes to train
# - memory_size, unsigned int: capacity of the replay buffer
# - visualize, bool: set true for actual screen displaying
# - architecture, string: specifiy architecture/network-model to use
# - xy_grid, unsigned int: "discretizes" the action spaces into NxN [x,y]-pairs
# - epsilon, unsigned int: the decay rate for the decaying epsilon greedy policy (epsilon ~ decayTime)
# - map_name, string: map/scenario to play
# - step_multiplier, unsigned int: how many game steps per agent step
def main(argv):
    del argv
    print(100 * "=")
    logging.info("Learning rate: {}".format(FLAGS.learning_rate))
    logging.info("Gamma: {}".format(FLAGS.gamma))
    logging.info("Batch Size: {}".format(FLAGS.batch_size))
    logging.info("Target Update Rate: {}".format(FLAGS.target_update))
    logging.info("Num Episodes: {}".format(FLAGS.epochs))
    logging.info("Memory Size: {}".format(FLAGS.memory_size))
    logging.info("Visualize: {}".format(FLAGS.visualize))
    logging.info("Architecture: {}".format(FLAGS.architecture))
    logging.info("Amount of xy coordinate pairs: {}".format(FLAGS.xy_factor**2))
    logging.info("Epsilon Decay value: {}".format(FLAGS.epsilon))
    logging.info("Chose map: {}".format(FLAGS.map_name))
    logging.info("Working directory {}".format(FLAGS.path))
    logging.info("Experiment name {}".format(FLAGS.name))
    logging.info("Imitation length {}".format(FLAGS.imitation_length))
    logging.info("Step multiplier {}".format(FLAGS.step_multiplier))
    print(100 * "=")

    flag_info = {
        "learning_rate" : FLAGS.learning_rate,
        "gamma": FLAGS.gamma,
        "batch_size" : FLAGS.batch_size,
        "target_update_rate" :FLAGS.target_update,
        "num_episodes" : FLAGS.epochs,
        "memory_size" : FLAGS.memory_size,
        "architecture" : FLAGS.architecture,
        "xy_factor" : FLAGS.xy_factor**2,
        "epsilon_decay" : FLAGS.epsilon,
        "map" : FLAGS.map_name,
        "imitation_length" : FLAGS.imitation_length,
        "step_multiplier" : FLAGS.step_multiplier
        }


    # save hyper parameter in csv
    hyper_path = FLAGS.path + "/experiment/" + FLAGS.name + "/"+ "hyper.csv"
    Path((FLAGS.path + "/experiment/" + FLAGS.name)).mkdir(parents=True, exist_ok=True)
    with open(hyper_path, "w") as f:
        writer = csv.writer(f)
        for key,value in flag_info.items():
            writer.writerow([key, value])

    env = gym.make('SC2MoveToBeacon-v0')
    print(env._states)
    print(env._actions)

    exit()

    for e in range(1, FLAGS.epochs + 1):

        observation = env.reset()

        while(True):

            action = agent.policy()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished")
                break











if __name__ == '__main__':
    ## contains hyperparameter input
    FLAGS = flags.FLAGS
    cwd = os.getcwd()
    flags.DEFINE_float("learning_rate", 0.0001, "Learning rate" )
    flags.DEFINE_float("gamma", 0.99 , "Gamma")
    flags.DEFINE_integer("batch_size", 32 , "Batch Size")
    flags.DEFINE_integer("target_update", 5 , "Update Target Network every N episodes")
    flags.DEFINE_integer("epochs", 10000 , "Amount of episodes")
    flags.DEFINE_integer("memory_size", 1000 , "Capacity of the ReplayBuffer")
    flags.DEFINE_bool("visualize", False, "Visualize pysc2 client")
    flags.DEFINE_string("architecture", "PytorchTutorialDQN", "Architecture to use for experiments")
    flags.DEFINE_integer("xy_factor", 5 , "shrink possible xy coordinates to N x N possible pairs")
    flags.DEFINE_integer("epsilon", 20000 , "epsilon epsilon decay")
    flags.DEFINE_string("map_name", "MoveToBeacon" , "Name of the map")
    flags.DEFINE_integer("step_multiplier", 1 , "specifiy step multiplier. 16 = ~1s game time")
    flags.DEFINE_string("path",cwd , "specify working directory for saving models and csv logs")
    flags.DEFINE_string("name", "default", "specify name for experiment")
    flags.DEFINE_integer("imitation_length", 5, "specifiy length of perfect phase to fill buffer with good memories")
    app.run(main)
