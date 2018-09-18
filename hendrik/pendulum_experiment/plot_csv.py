#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('rewards_per_epoch.csv', usecols=['reward'])

plt.figure()
plt.plot(df)
plt.show()
