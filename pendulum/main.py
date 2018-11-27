#!/usr/bin/env python3

import gym
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

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

env = gym.make('CartPole-v0').unwrapped
env.reset()

agent = PendulumAgent(agent_specs)
agent.set_learning_mode()
num_episodes = 50
for i_episode in range(num_episodes):

    # Initialize the environment and state
    state, reward, done, info = env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        action = agent.policy(state, reward, done, info)
        print(action)
        _, reward, done, info = env.step(action)
        last_screen = current_screen
        current_screen = get_screen(env)
        print(current_screen.shape)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        dict_agent_report = agent.evaluate(next_state, reward, done, info)

        # Move to the next state
        state = next_state

        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            break


print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
