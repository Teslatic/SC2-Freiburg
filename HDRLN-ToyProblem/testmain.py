#!/usr/bin/env python3

# python imports
from absl import app, flags
from itertools import count

# gym imports
import gym
import gym_toyproblems

# custom imports
from assets.agents.ToyAgent import ToyAgent
from assets.plotting.plotter import Plotter
from specs.agent_specs import agent_specs
from specs.env_specs import cartpole_specs
from assets.splash.squidward import print_squidward
from assets.helperFunctions.FileManager import FileManager

def main(argv):
    # print_squidward()

    # FileManager: Save specs and create experiment
    fm = FileManager()
    try:
        spec_summary = fm.load_spec_summary(FLAGS.specs)
        fm.change_cwd(spec_summary["ROOT_DIR"])
    except:
        print("Loading specs/model failed. Have you selected the right path?")
        exit()
    fm.create_test_file()

    # show_extracted_screen(get_screen(env))
    plotter = Plotter()

    # No FileManager yet
    agent = ToyAgent(spec_summary)
    agent.DQN.load(FLAGS.model)
    agent.set_testing_mode()

    list_reward_per_episode = []

    env = gym.make(agent.gym_string).unwrapped

    num_episodes = int(spec_summary["TEST_EPISODES"])
    for e in range(num_episodes):
        reward = 0
        done = False
        info = None

        # Initialize the environment and state
        state = env.reset()

        reward_cnt = 0
        try:
            print("Episode {} | Last reward: {}".format(e, list_reward_per_episode[-1]))
        except:
            pass

        for t in count():
            # Select and perform an action
            action = agent.policy(state, reward, done, info)
            next_state, reward, done, info = env.step(action)

            # if spec_summary['VISUALIZE']=="True":
            env.render()
            reward_cnt += reward
            if not done:
                pass
            else:
                next_state = None
                agent.episodes += 1
                plotter.episode_durations.append(reward_cnt)
                # plotter.plot_durations()

            # Store the transition in memory
            test_report = agent.evaluate(next_state, reward, done, info)
            fm.log_test_reports(test_report)

            # Move to the next state
            state = next_state

            if done:
                list_reward_per_episode.append(reward_cnt)
                dict_test_report = {"RewardPerEpisode": list_reward_per_episode }
                fm.log_test_reports(dict_test_report)
                break
    print('Testing complete')
    env.close()
    plotter.close()

if __name__ == "__main__":
    # Arg parsing for model and specs paths
    FLAGS = flags.FLAGS
    flags.DEFINE_string("specs", None, "path to spec summary")
    flags.DEFINE_string("model", None, "path to pytorch model")
    app.run(main)
