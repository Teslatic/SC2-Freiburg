#!/usr/bin/env python3
"""
Hendrik Vloet
Copyright (C) 2018 Hendrik Vloet
Public Domain
"""
# ______________________________________________________________________________

## @package plotter
#
#  serves to plot a given csv containing the reward per epoch and cumulative
#  reward progression over time

# python imports
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import argparse

parser = argparse.ArgumentParser(description="specify patho to csv file")
parser.add_argument("--path", type=str, help="path to csv directory")


args = parser.parse_args()
reward_csv_path = args.path + "reward_per_epoch.csv"
coordinates_csv_path = args.path + "coordinates.csv"


""" Plot rewards """

df = pd.read_csv(reward_csv_path)
fig, axes = plt.subplots(nrows=1, ncols=len(df.columns))
colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
for idx,column, in enumerate(df):
    df[column].plot(ax=axes[idx], color=colors[idx], title='Reward progression').set_title(column)
fig.text(0.04, 0.5, 'Reward', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Epoch', ha='center')
plt.savefig(reward_csv_path.replace(".csv", "_progression.png"), dpi = 1200)


""" Plot chosen coordinates heatmap """

df = pd.read_csv(coordinates_csv_path)
fig, _  = plt.subplots()
ax = df.plot.hexbin('x', 'y', gridsize=25, cmap="Blues")
plt.title("Heatmap of chosen (x,y) pairs", pad = 30)
plt.gca().invert_yaxis()
ax.xaxis.tick_top()
plt.savefig(coordinates_csv_path.replace(".csv", "_heatmap.png"), dpi = 1200)
