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
import pandas as pd

# torch imports
import torch


# import architectures
from Architectures import PytorchTutorialDQN,  FullyConv

# import Agent
from TestAgency import TestAgent


def main(argv):
    del argv
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # read in hyperparameter csv file and convert it into dict
    hyper_params = pd.read_csv(FLAGS.path + "hyper.csv",  header=None, index_col=0).to_dict()

    # for some reason, the dictionary created by the read-in, is a dict in a dict with one key 1
    hyper_params = hyper_params[1]

    # convert dictionary into namedtuple to make it like the flags
    Hyper_tuple = namedtuple('Hyper_tuple', sorted(hyper_params))
    hyper_tuple = Hyper_tuple(**hyper_params)

    print(100 * "=")
    print("Found following hyperparameter:")
    for key in hyper_tuple._fields:
        print(key, getattr(hyper_tuple, key))
    print(100 * "=")

    if (hyper_tuple.architecture == "PytorchTutorialDQN"):
        architecture = PytorchTutorialDQN(hyper_tuple)
    if (hyper_tuple.architecture == "FullyConv"):
        architecture = FullyConv(hyper_tuple)
    ## agent object
    agent = TestAgent(architecture, FLAGS, hyper_tuple, device)
    agent.test()



if __name__ == '__main__':
    FLAGS = flags.FLAGS
    cwd = os.getcwd()

    flags.DEFINE_string("path", cwd, "specifiy model directory")
    flags.DEFINE_integer("epochs", 100 , "Amount of test episodes")
    flags.DEFINE_bool("visualize", True, "Visualize pysc2 client")
    flags.DEFINE_integer("step_multiplier", 1 , "specifiy step multiplier. 16 = ~1s game time")
    
    app.run(main)
