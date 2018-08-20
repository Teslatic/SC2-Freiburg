#!/usr/bin/env python3
# pysc2 modules
from pysc2.agents import  random_agent, scripted_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

### torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



# custom imports
from BaseAgent import BaseAgent
from AtariNet import ReplayBuffer

# normal python modules
import random
import argparse
from collections import namedtuple

# CONSTANTS
SCREEN_DIM = 84
MINIMAP_DIM = 64
GAMESTEPS = None # 0 = unlimited game time, None = map default
STEP_MULTIPLIER = 32 # 16 = 1s game time
VISUALIZE = True
BATCH_SIZE = 32


# Initalizing
replay_memory_size = 10
replay_memory = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


'''
helpers
'''
def setup_agent():
    agent = BaseAgent(screen_dim = SCREEN_DIM, minimap_dim = MINIMAP_DIM, batch_size=BATCH_SIZE)
    # agent = scripted_agent.MoveToBeacon()
    # players = [ sc2_env.Agent(sc2_env.Race.terran),
    #             sc2_env.Bot(sc2_env.Race.random,
    #             sc2_env.Difficulty.very_easy)]
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    agent_interface = features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=SCREEN_DIM,
                                                    minimap=MINIMAP_DIM),
            # rgb_dimensions = features.Dimensions(screen=SCREEN_DIM,
            #                                         minimap=MINIMAP_DIM),
            # action_space = actions.ActionSpace.RGB,
            use_feature_units = True)

    return agent, players, agent_interface

def setup_env(agent, players, agent_interface):
    env = sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=players,
            agent_interface_format=agent_interface,
            step_mul=STEP_MULTIPLIER,
            game_steps_per_episode=GAMESTEPS,
            visualize=VISUALIZE)
    return env


'''
main function
'''
def main(unused_argv):
    agent, players, agent_interface = setup_agent()
    memory = ReplayBuffer(1000)

    try:
        while True:
            with  setup_env(agent,players,agent_interface) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()
                while True:
                    state = timesteps[0].observation["feature_screen"]["player_relative"]
                    action = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(action)
                    reward = timesteps[0].reward
                    next_state = timesteps[0].observation["feature_screen"]["player_relative"]

                    memory.push(state, action, reward, next_state, timesteps[0].step_type)

                    if len(memory) >= BATCH_SIZE:
                        batch = memory.sample(BATCH_SIZE)
                        agent.train(batch)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
