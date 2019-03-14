import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from specs.agent_specs import agent_specs


class Plotter():
    """
    A simple plotter class.
    """
    def __init__(self):
        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display
        plt.ion()
        self.episode_durations = []

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training {}...'.format(agent_specs["AGENT_TYPE"]))
        plt.xlabel('Episode')
        plt.ylabel('RewardPerEpisode')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 50:
            means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(49), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def plot_diff(self, diff_screen):
        plt.figure(3)
        plt.imshow(np.transpose(diff_screen.squeeze(0),(1,2,0)), interpolation='none')
        plt.title('difference screen')
        # plt.pause(0.0001)  # pause a bit so that plots are updated

        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

            # plt.show()

    def show_extracted_screen(self):
        plt.figure()
        # print(get_screen(env).squeeze(0).shape)
        plt.imshow(np.transpose(screen.squeeze(0),(1,2,0)), interpolation='none')
        plt.title('Example extracted screen')
        plt.show()

    def close(self):
        """
        Clean closing of plots.
        """
        plt.ioff()
        plt.show()
