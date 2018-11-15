#!/usr/bin/env python3

# gym imports
import gym
import gym_ghost


# python imports
from absl import app
from absl import flags
from absl import logging
import time

# custom imports
from ReplayBuffer import ReplayBuffer
from env_specs import mv2beacon_specs
from agent_specs import mv2beacon_specs
from squidward import print_squidward

from CompassAgent import CompassAgent

# pysc2 imports
from pysc2.lib import actions

def main(argv):
    del argv
    env = gym.make("sc2-v0")
    agent = CompassAgent(mv2beacon_specs)
    # print_squidward()
    env.setup(mv2beacon_specs)

    memory = ReplayBuffer

    obs, reward, done, info  = env.reset()

    while(True):
        action = agent.policy(obs)

        if (action is 'reset'):
            print_ts("About to reset the environment")
            next_obs = env.reset()
            print_ts("Environment reset. Episode finished.")
            end_time = time.time()
            print_ts("Episode took {} seconds.".format(end_time-start_time))
            start_time = time.time()
            # raise Exception("Test for weight saving")
        else:
            # Peforming selected action
            next_obs = env.step(action)

        if not actual_obs.first() and not actual_obs.last():  # make one step
            # Saving the episode data. Pushing the information onto the memory.
            agent.store_transition(next_obs[0])
            # Optimize the agent
            if agent.get_memory_length() >= 100:
                agent.optimize()

                # Print actual status information
                if not agent.silentmode:
                    agent.print_status()

        if agent.logging:
            agent.log()

        actual_obs = next_obs[0]
        obs, reward, done, info = env.step('up')
        time.sleep(0.1)

        print(reward)
if __name__=="__main__":
    app.run(main)


