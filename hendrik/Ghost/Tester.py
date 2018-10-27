#!/usr/bin/env python3
"""
Hendrik Vloet
Copyright (C) 2018 Hendrik Vloet
Public Domain
"""
# ______________________________________________________________________________



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


# import architectures
from Architectures import PytorchTutorialDQN, ConvNet, FullyConv

# import Agent
from TestAgency import TestAgent


def main(argv):
    del argv
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (FLAGS.architecture == "PytorchTutorialDQN"):
        architecture = PytorchTutorialDQN(FLAGS)
    if (FLAGS.architecture == "ConvNet"):
        architecture = ConvNet(FLAGS)
    if (FLAGS.architecture == "FullyConv"):
        architecture = FullyConv(FLAGS)
    ## agent object
    agent = TestAgent(architecture, FLAGS, device)
    agent.test()



if __name__ == '__main__':
    FLAGS = flags.FLAGS
    cwd = os.getcwd()

    flags.DEFINE_string("path", cwd, "specifiy model directory")
    flags.DEFINE_string("map_name", "MoveToBeacon" , "Name of the map")
    flags.DEFINE_integer("epochs", 100 , "Amount of test episodes")
    flags.DEFINE_integer("xy_grid", 5 , "shrink possible xy coordinates to N x N possible pairs")
    flags.DEFINE_bool("visualize", False, "Visualize pysc2 client")
    flags.DEFINE_string("architecture", "PytorchTutorialDQN", "Architecture to use for experiments")
    flags.DEFINE_integer("step_multiplier", 0 , "specifiy step multiplier. 16 = ~1s game time")

    app.run(main)
