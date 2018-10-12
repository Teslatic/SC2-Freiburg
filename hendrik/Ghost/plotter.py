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

## parser object to gather the path
parser = argparse.ArgumentParser(description="specify patho to csv file")
parser.add_argument("--path", type=str, help="path to csv file")

## parsing objects to args
args = parser.parse_args()
## extracting path
path = args.path

## pandas dataframe
df = pd.read_csv(path)

## plotting setup
fig, axes = plt.subplots(nrows=1, ncols=len(df.columns))

## colortable for multiple subplots
colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

for idx,column, in enumerate(df):
    df[column].plot(ax=axes[idx], color=colors[idx], title='Reward progression').set_title(column)

fig.text(0.04, 0.5, 'Reward', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Epoch', ha='center')

plt.savefig(path.replace(".csv", "_progression.png"))


