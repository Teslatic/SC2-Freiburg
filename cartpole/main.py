#!/usr/bin/env python3

# python imports
from absl import app
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
        fm.create_experiment(agent_specs["EXP_NAME"])  # Automatic cwd switch
        fm.save_specs(agent_specs, cartpole_specs)
    except:
        print("Creating eperiment or saving specs failed.")
        exit()
    fm.create_train_file()

    # show_extracted_screen(get_screen(env))
    plotter = Plotter()

    # No FileManager yet
    agent = CartPoleAgent(agent_specs)
    agent.set_learning_mode()

    env = gym.make('CartPole-v0').unwrapped

    num_episodes = cartpole_specs["EPISODES"]
    for e in range(num_episodes):

        # Initialize the environment and state
        state, reward, done, info = env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen

        if e%50==0 and e!=0:
            agent.save_model(fm.get_cwd())

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
                # agent.update_target_network()
                plotter.episode_durations.append(t + 1)
                plotter.plot_durations()

            print(e, t, reward, action, len(agent.DQN.memory), agent.epsilon)
            # print(agent.DQN.state_q_values)
            # Store the transition in memory
            train_report = agent.evaluate(next_state, reward, done, info)
            fm.log_training_reports(train_report)

            # Move to the next state
            state = next_state

            if done:
                break

        agent.update_target_network()

    agent.save_model(fm.get_cwd())
    print('Training complete')
    env.close()
    plotter.close()

if __name__ == "__main__":
    # No flags for arg parsing defined yet.
    app.run(main)
