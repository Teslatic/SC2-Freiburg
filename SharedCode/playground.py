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
from agent_specs import compassagent_specs, gridagent_specs
from CompassAgent import CompassAgent
from GridAgent import GridAgent
from squidward import print_squidward
from assets.helperFunctions.timestamps import print_timestamp as print_ts


# pysc2 imports
from pysc2.lib import actions

def main(argv):
    del argv
    print_squidward()
    agent = CompassAgent(compassagent_specs)
    # agent = GridAgent(gridagent_specs)
    env = gym.make("sc2-v0")
    obs, reward, done, info  = env.setup(mv2beacon_specs)

    while(True):
        action = agent.policy(obs, reward, done, info)
        print("Selected action is: {}".format(action))
        if (action is 'reset'):
            print_ts("About to reset the environment")
            obs, reward, done, info = env.reset()
            print_ts("Environment reset. Episode finished.")
            # end_time = time.time()
            # print_ts("Episode took {} seconds.".format(end_time-start_time))
            # start_time = time.time()
            # raise Exception("Test for weight saving")
        else:
            # Peforming selected action
            # next_obs = env.step(action)
            # obs, reward, done, info = env.step('up')
            obs, reward, done, info = env.step(action)
            # time.sleep(0.1)
            # Saving the episode data. Pushing the information onto the memory.
            agent.store_transition(obs, reward)
            # Optimize the agent
            agent.optimize()

                # # Print actual status information
                # if not agent.silentmode:
                #     agent.print_status()

        # if agent.logging:
        #     agent.log()




        print(reward)
if __name__=="__main__":
    app.run(main)
