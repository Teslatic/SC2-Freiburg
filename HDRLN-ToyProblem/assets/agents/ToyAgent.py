# python imports
import numpy as np
import pandas as pd
from sys import getsizeof

# custom imports
from assets.RL.DQN_module import DQN_module
from assets.helperFunctions.timestamps import print_timestamp as print_ts
# from assets.helperFunctions.FileManager import create_experiment_at_main

# from assets.agents.smart_actions import SMART_ACTIONS_PENDULUM as SMART_ACTIONS

COMMON_ACTION_SPACE = np.array([0,1,2])
ACROBOT_INPUT_DIM = 6
CARTPOLE_INPUT_DIM = 4
PENDULUM_INPUT_DIM = 3

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)


class ToyAgent():
    """
    This is a simple agent that uses an PyTorch DQN_module as Q value
    approximator. Current implemented features of the agent:
    - Simple initializing with the help of an agent_specs.
    - Policy switch between imitation and epsilon greedy learning session.
    - Storage of experience into simple Experience Replay Buffer
    - Intermediate saving of model weights

    To be implemented:
    - Saving the hyperparameter file of the experiments.
    - Storing the DQN model weights in case of Runtime error.
    """
    # ##########################################################################
    # Initializing the agent
    # ##########################################################################

    def __init__(self, agent_specs):
        """
        steps_done: Total timesteps done in the agents lifetime.
        timesteps:  Timesteps performed in the current episode.
        choice:     Choice of epsilon greedy method (None if supervised)
        loss:       Action loss
        hist:       The history buffer for more complex minigames.
        """
        self.unzip_hyperparameter_file(agent_specs)
        self.dim_actions = 3
        self.episodes = 1
        self.choice = None  # Choice of epsilon greedy
        self.episode_reward_env = 0
        self.episode_reward_shaped = 0
        self.loss = 0  # Action loss
        self.reward = 0
        self.steps_done = 0  # total_timesteps
        self.steps = 0
        self.timesteps = 0  # timesteps in the current episode
        self.setup_dqn()
        self.done = False

        # counter for book keepping
        self.shaped_reward_cumulative = 0
        self.shaped_reward_per_episode = 0
        self.pysc2_reward_cumulative = 0

        # lists for book keeping
        self.list_shaped_reward_per_episode = []
        self.list_shaped_reward_cumulative = []
        self.list_epsilon_progression = []
        self.list_loss_per_episode = []
        self.list_loss_mean = []
        self.list_pysc2_reward_per_episode = []
        self.list_pysc2_reward_cumulative = []

    def get_input_dim(self):
        if self.agent_type == 'pendulum':
            self.gym_string = 'gym-toy-pendulum-v0'
            return PENDULUM_INPUT_DIM
        elif self.agent_type == 'cartpole':
            self.gym_string = 'gym-toy-cartpole-v0'
            return CARTPOLE_INPUT_DIM
        elif self.agent_type == 'acrobot':
            self.gym_string = 'gym-toy-acrobot-v0'
            return ACROBOT_INPUT_DIM

    def setup_dqn(self):
        self.input_dim = self.get_input_dim()
        self.smart_actions = COMMON_ACTION_SPACE
        print_ts("Setup DQN of CartPoleAgent")
        self.DQN = DQN_module(self.batch_size,
                              self.gamma,
                              self.history_length,
                              self.size_replaybuffer,
                              self.optim_learning_rate,
                              self.input_dim)
        self.device = self.DQN.device
        print_ts("DQN module has been initalized")

    def unzip_hyperparameter_file(self, agent_specs):
        """
        Unzipping and writing the hyperparameter file into member variables.
        """
        self.gamma = float(agent_specs['GAMMA'])
        self.optim_learning_rate = float(agent_specs['OPTIM_LR'])
        self.batch_size = int(agent_specs['BATCH_SIZE'])
        self.target_update_period = int(agent_specs['TARGET_UPDATE_PERIOD'])
        self.history_length = int(agent_specs['HIST_LENGTH'])
        self.size_replaybuffer = int(agent_specs['REPLAY_SIZE'])
        # self.device = agent_specs['DEVICE']
        # self.silentmode = agent_specs['SILENTMODE']
        # self.logging = agent_specs['LOGGING']
        self.supervised_episodes = int(agent_specs['SUPERVISED_EPISODES'])
        self.patience = int(agent_specs['PATIENCE'])

        # epsilon_file = agent_specs['EPSILON_FILE']
        self.eps_start = float(agent_specs['EPS_START'])
        self.eps_end = float(agent_specs['EPS_END'])
        self.eps_decay = int(agent_specs['EPS_DECAY'])
        # epsilon in current timestep
        self.epsilon = 0

        self.mode = agent_specs['MODE']
        self.agent_type = agent_specs['AGENT_TYPE']


        self.grid_dim_x = int(agent_specs['GRID_DIM_X'])
        self.grid_dim_y = int(agent_specs['GRID_DIM_Y'])

        self.exp_name = agent_specs['EXP_NAME']


    # ##########################################################################
    # Action Selection
    # ##########################################################################

    def prepare_timestep(self, obs, reward, done, info):
        """
        timesteps:
        """
        # from PYSC2 base class
        self.steps += 1
        self.reward += reward

        # Current episode
        self.timesteps += 1
        self.episode_reward_env += reward
        # TODO(vloeth): extend observation to full pysc2 observation
        # self.available_actions = obs.observation.available_actions

        # Unzip the observation tuple
        self.state = obs
        # mode switch
        if (self.episodes > self.supervised_episodes) and \
            (not self.mode == 'testing'):
            self.set_learning_mode()


    def policy(self, obs, reward, done, info):
        """
        Choosing an action
        """
        # Set all variables at the start of a new timestep
        self.prepare_timestep(obs, reward, done, info)

        # Action selection for regular step
        if self.mode == 'supervised':
            # For the first n episodes learn on forced actions.
            self.action, self.action_idx = self.supervised_action()
        if self.mode == 'learning':
            # Choose an action according to the policy
            self.action, self.action_idx = self.epsilon_greedy_action()
        if self.mode == 'testing':
            self.action, self.action_idx = self.pick_action()

        # self.done = False
        # if self.done:  # End episode in last step
        #     if self.mode != 'testing':
        #         self.update_target_network()

        return self.action

    def pick_action(self):
        action_q_values = self.DQN.predict_q_values(self.state)
        # print("action q values", action_q_values)
        # Beste Action bestimmen
        best_action_numpy = action_q_values.detach().cpu().numpy()
        action_idx = np.argmax(best_action_numpy)
        best_action = self.smart_actions[action_idx]
        chosen_action = best_action
        return chosen_action, action_idx

    def epsilon_greedy_action(self):
        """
        chooses an action according to the current policy
        returns the chosen action id with x,y coordinates and the index of the
        action with respect to the SMART_ACTIONS constant. The additional index
        is used for the catergorical labeling of the actions as an
        "intermediate" solution for a restricted action space.
        """
        self.choice = self.epsilon_greedy()
        if self.choice == 'random':
            action_idx = np.random.randint(self.dim_actions)
            chosen_action = self.smart_actions[action_idx]
        else:
            chosen_action, action_idx = self.pick_action()
        return chosen_action, action_idx

    def epsilon_greedy(self):
        """
        returns a string in order to determine if the next action choice is
        going to be random or according to an decaying epsilon greeedy policy
        """
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) \
            * np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        return np.random.choice(['random', 'greedy'],
                                p=[self.epsilon, 1-self.epsilon])

    # ##########################################################################
    # Evaluate a timestep
    # ##########################################################################

    def evaluate(self, obs, reward, done, info):
        """
        A generic wrapper, thaprintt contains all agent operations which are used
        after finishing a timestep.

        Retuns a dictionary with the following information:
            - Shaped reward per episode
            - Shaped reward cumulative
            - Mean loss per episode
            - epsilon progression per episode
        """
        try:

            if self.mode != 'testing':
                # Saving the episode data. Pushing the information onto the memory.
                self.store_transition(obs, reward)

                # Optimize the agent
                self.optimize()

            # collect reward, loss and epsilon information as dictionary
            # agent_report = self.collect_report(obs, reward, done)
            return None
        except KeyboardInterrupt:
            self.save_model(emergency=True)
        return agent_report

    def set_learning_mode(self):
        """
        Set the agent to learning mode. The agent will perform
        actions according to an epsilon-greedy policy.
        """
        self.mode = "learning"

    def set_supervised_mode(self):
        """
        Set the agent to supervised mode. The agent will perform
        actions which are generating valuable experience.
        """
        self.mode = "supervised"

    def set_testing_mode(self):
        """
        Set the agent to testing mode. The agent will perform only
        aexploiting actions without any randomness.
        """
        self.mode = "testing"

    def collect_report(self, obs, reward, done):
        """
        Retuns a dictionary with the following information:
            - Shaped reward per episode
            - Shaped reward cumulative
            - Mean loss per episode
            - epsilon progression per episode
        """
        self.shaped_reward_cumulative += reward
        self.shaped_reward_per_episode += reward


        if self.get_memory_length() >= self.batch_size * self.patience:
            self.list_loss_per_episode.append(self.loss.data[0])
        else:
            self.list_loss_per_episode.append(self.loss)


        if done:
            self.list_shaped_reward_cumulative.append(self.shaped_reward_cumulative)
            self.list_shaped_reward_per_episode.append(self.shaped_reward_per_episode)

            self.list_loss_mean.append(np.mean(self.list_loss_per_episode))

            # score observation per episode from pysc2 is appended
            self.list_pysc2_reward_per_episode.append(obs[7])

            # cumulative pysc2 reward
            self.pysc2_reward_cumulative += obs[7]
            self.list_pysc2_reward_cumulative.append(self.pysc2_reward_cumulative)


            # if self.episodes > self.supervised_episodes:
            self.list_epsilon_progression.append(self.epsilon)
            # else:
                # self.list_epsilon_progression.append(self.eps_start)

            dict_agent_report = {
                 "ShapedRewardPerEpisode": self.list_shaped_reward_per_episode,
                 "ShapedRewardCumulative":self.list_shaped_reward_cumulative,
                 "Pysc2RewardPerEpisode": self.list_pysc2_reward_per_episode,
                 "Pysc2RewardCumulative": self.list_pysc2_reward_cumulative,
                 "MeanLossPerEpisode": self.list_loss_mean,
                 "Epsilon": self.list_epsilon_progression,
                }
            print("Last epsilon: {}".format(self.list_epsilon_progression[-1]))
            return dict_agent_report



    def store_transition(self, next_obs, reward):
        """
        Save the actual information in the history. As soon as there has been
        enough data, the experience is sampled from the replay buffer.
        TODO: Maybe using uint8 for storing into ReplayBuffer
        """
        self.reward = reward
        self.next_state = next_obs

        self.DQN.memory.push(self.state,
                             [self.action_idx],
                             self.reward,
                             self.next_state)
    # ##########################################################################
    # DQN module wrappers
    # ##########################################################################

    def get_memory_length(self):
        """
        Returns the length of the ReplayBuffer
        """
        return len(self.DQN.memory)

    def optimize(self):
        """
        Optimizes the DQN_module on a minibatch
        """
        if self.get_memory_length() >= self.batch_size * self.patience:
            self.loss = self.DQN.optimize()

    # ##########################################################################
    # Ending Episode
    # ##########################################################################

    def update_target_network(self):
        """
        Transferring the estimator weights to the target weights and resetting
        the agent.
        """
        if self.episodes % self.target_update_period == 0:
            print_ts("About to update")
            self.DQN.update_target_net()

    def reset(self):
        """
        Resetting the agent --> More explanation
        """
        super(Move2BeaconAgent, self).reset()
        self.timesteps = 0
        self.episode_reward_env = 0
        self.episode_reward_shaped = 0

        self.shaped_reward_per_episode = 0
        self.list_loss_per_episode = []

    # ##########################################################################
    # Print status information of timestep
    # ##########################################################################
        # Path((self.exp_path + "/model")).mkdir(parents=True, exist_ok=True)

    def save_model(self, exp_path, emergency=False):
        if emergency:
            print("KeyboardInterrupt detected, saving last model!")
            save_path = exp_path + "/model/emergency_model_{}.pt".format(self.episodes)
        else:
            save_path = exp_path + "/model/model_{}.pt".format(self.episodes)

        # Path((self.exp_path + "/model")).mkdir(parents=True, exist_ok=True)
        self.DQN.save(save_path)
        print_ts("Model has been saved at {}".format(save_path))

    def _load_model(self):
        load_path = self.exp_path + "/model/emergency_model.pt"

        self.DQN.load(load_path)
