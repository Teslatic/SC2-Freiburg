#!/usr/bin/env python3

# python imports
from absl import app, flags
from itertools import count

# gym imports
import gym

# custom imports
from assets.helperFunctions.screen_extraction import get_screen
from assets.agents.CartPoleAgent import CartPoleAgent
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
    agent = CartPoleAgent(agent_specs)
    agent.set_testing_mode()

    env = gym.make('CartPole-v0').unwrapped

    num_episodes = int(spec_summary["TEST_EPISODES"])
    for e in range(num_episodes):

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
                plotter.episode_durations.append(t + 1)
                plotter.plot_durations()

            print(e, t, reward, action)
            # Store the transition in memory
            test_report = agent.evaluate(next_state, reward, done, info)
            fm.log_test_reports(test_report)

            # Move to the next state
            state = next_state

            if done:
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
