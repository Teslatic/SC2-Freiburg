#!/usr/bin/env python3

import gym
import numpy as np

from itertools import count

from assets.helperFunctions.screen_extraction import get_cart_location, get_screen, resize
from assets.agents.PendulumAgent import PendulumAgent
from assets.plotting.plotter import Plotter
# custom imports
from specs.agent_specs import agent_specs
from specs.env_specs import mv2beacon_specs
from assets.helperFunctions.initializingHelpers import setup_agent
# from assets.helperFunctions.FileManager import FileManager

import torch

env = gym.make('CartPole-v0').unwrapped
env.reset()

# show_extracted_screen(get_screen(env))

agent = PendulumAgent(agent_specs)
agent.set_learning_mode()
num_episodes = 500

plotter = Plotter()

for i_episode in range(num_episodes):

    # Initialize the environment and state
    state, reward, done, info = env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        action = agent.policy(state, reward, done, info)
        _, reward, done, info = env.step(action)
        last_screen = current_screen
        current_screen = get_screen(env)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
            agent.episodes += 1
            agent.update_target_network()
            plotter.episode_durations.append(t + 1)
            plotter.plot_durations()
            # break
        print(i_episode, t, reward, action, len(agent.DQN.memory), agent.epsilon)


        # Store the transition in memory
        dict_agent_report = agent.evaluate(next_state, reward, done, info)
        # Move to the next state
        state = next_state

        if done:
            break

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
