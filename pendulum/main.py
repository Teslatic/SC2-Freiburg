#!/usr/bin/env python3

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from assets.helperFunctions.screen_extraction import get_cart_location, get_screen, resize
from assets.agents.PendulumAgent import PendulumAgent
# custom imports
from specs.agent_specs import agent_specs
from specs.env_specs import mv2beacon_specs
from assets.helperFunctions.initializingHelpers import setup_agent
# from assets.helperFunctions.FileManager import FileManager

import torch
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_diff(diff_screen):
    plt.figure(3)
    plt.imshow(np.transpose(diff_screen.squeeze(0),(1,2,0)), interpolation='none')
    plt.title('difference screen')
    plt.pause(0.0001)  # pause a bit so that plots are updated

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    # plt.show()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

env = gym.make('CartPole-v0').unwrapped


env.reset()
plt.figure()
# print(get_screen(env).squeeze(0).shape)
plt.imshow(np.transpose(get_screen(env).squeeze(0),(1,2,0)), interpolation='none')
plt.title('Example extracted screen')
plt.show()

episode_durations = []

agent = PendulumAgent(agent_specs)
agent.set_learning_mode()
num_episodes = 500

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
            # plot_diff(next_state)


        else:
            next_state = None
            agent.episodes += 1
            agent.update_target_network()
            episode_durations.append(t + 1)
            plot_durations()
            # break
        print(i_episode, t, reward, agent.epsilon, action)


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
