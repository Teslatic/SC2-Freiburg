#!/usr/bin/env python3
"""
Hendrik Vloet
Copyright (C) 2018 Hendrik Vloet
Public Domain
"""
# ______________________________________________________________________________

## \mainpage StarCraft 2 Reinforcement Learning Project
#  Albert Ludwigs Universität Freiburg  
#  Project at the chair of Neurorobotics  
#  Supervisor: Prof. Dr. J. Bödecker  
#  Students: Nico Ott, Hendrik Vloet  


# normal python imports
from absl import app
from absl import flags
from absl import logging
import random
import numpy as np
import math
from itertools import count
from collections import namedtuple

# torch imports
import torch


# import architectures
from Architectures import PytorchTutorialDQN

# import Agent
from Agency import BaseAgent

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
    logging.info("Amount of xy coordinate pairs: {}".format(FLAGS.xy_grid**2))
    logging.info("Epsilon Decay value: {}".format(FLAGS.epsilon))
    logging.info("Chose map: {}".format(FLAGS.map_name))
    print(100 * "=")

    ## device information: run on GPU if possible, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (FLAGS.architecture == "PytorchTutorialDQN"):
        ## architecture objects: contains the pytorch network
        architecture = PytorchTutorialDQN(FLAGS)
    ## agent object
    agent = BaseAgent(architecture, FLAGS, device)
    agent.play()

if __name__ == '__main__':
    ## contains hyperparameter input
    FLAGS = flags.FLAGS

    flags.DEFINE_float("learning_rate", 0.0001, "Learning rate" )
    flags.DEFINE_float("gamma", 0.9 , "Gamma")
    flags.DEFINE_integer("batch_size", 32 , "Batch Size")
    flags.DEFINE_integer("target_update", 5 , "Update Target Network every N episodes")
    flags.DEFINE_integer("epochs", 10000 , "Amount of episodes")
    flags.DEFINE_integer("memory_size", 1000 , "Capacity of the ReplayBuffer")
    flags.DEFINE_bool("visualize", False, "Visualize pysc2 client")
    flags.DEFINE_string("architecture", "PytorchTutorialDQN", "Architecture to use for experiments")
    flags.DEFINE_integer("xy_grid", 5 , "shrink possible xy coordinates to N x N possible pairs")
    flags.DEFINE_integer("epsilon", 20000 , "epsilon epsilon decay")
    flags.DEFINE_string("map_name", "MoveToBeacon" , "Name of the map")
    flags.DEFINE_integer("step_multiplier", 0 , "specifiy step multiplier. 16 = ~1s game time")

    app.run(main)
