# python imports
import numpy as np

# custom imports
from pysc2.agents import base_agent
from assets.RL.DQN_module import DQN_module
from assets.helperFunctions.timestamps import print_timestamp as print_ts
from assets.actionspace.utils import setup_action_space

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)


class DQNBaseAgent(base_agent.BaseAgent):
    """
    This is a simple agent that uses an PyTorch DQN_module as Q value
    approximator. Current implemented features of the agent:
    - Simple initializing with the help of an agent_specs.
    - Policy switch between imitation and epsilon greedy learning session.
    - Storage of experience into simple Experience Replay Buffer
    - Intermediate saving of model weights
    """
    # ##########################################################################
    # Initializing
    # ##########################################################################

    def __init__(self, agent_specs):
        """
        1. Setup base agent functionality
        2. Unzip agent specs
        3. Setup crucial framework parameters
        4. Setup the action space
        5. Setup the DQN-module
        """
        self._setup_base_agent()
        self._unzip_agent_specs(agent_specs)
        self._setup_framework_parameters()

        self.action_space, self.dim_actions = setup_action_space(agent_specs)
        self.module_specs = self.zip_module_specs()
        self.setup_dqn(self.module_specs)

        # Initial mode switch
        if (self.episodes > self.supervised_episodes) and \
                    (not self.mode == 'testing'):
                    self.set_learning_mode()

    def _setup_base_agent(self):
        """
        Setting up the parameters from the PYSC2 base class.
        """
        super(DQNBaseAgent, self).__init__()

    def _unzip_agent_specs(self, agent_specs):
        """
        Unzipping and writing the hyperparameter file into member variables.
        exp_name: Name of create_experiment
        action_type: Used action type for the environment
        gamma: Discount factor for target calculation
        optim_learning_rate: Learning rate of the optimizer
        """
        self.exp_name = agent_specs['EXP_NAME']
        self.action_type = agent_specs['ACTION_TYPE']

        self.gamma = float(agent_specs['GAMMA'])
        self.optim_learning_rate = float(agent_specs['OPTIM_LR'])

        self.eps_start = float(agent_specs['EPS_START'])
        self.eps_decay = int(agent_specs['EPS_DECAY'])
        self.eps_end = float(agent_specs['EPS_END'])
        self.epsilon = 0

        self.batch_size = int(agent_specs['BATCH_SIZE'])
        self.size_replaybuffer = int(float(agent_specs['REPLAY_SIZE']))
        self.target_update_period = int(agent_specs['TARGET_UPDATE_PERIOD'])

        self.mode = agent_specs['MODE']
        self.supervised_episodes = int(agent_specs['SUPERVISED_EPISODES'])

        self.model_save_period = agent_specs['MODEL_SAVE_PERIOD']

    def _setup_framework_parameters(self):
        """
        Setting up all the parameters which are needed for the framework routine
        to run.
        """
        self.episodes = 1
        self.choice = None  # Choice of epsilon greedy
        self.episode_reward_env = 0
        self.episode_reward_shaped = 0
        self.loss = 0  # Action loss
        self.reward = 0
        self.steps_done = 0  # total_timesteps
        self.timesteps = 0  # timesteps in the current episode
        self.history_length = 1
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

    def setup_dqn(self, module_specs):
        # TODO(vloet): what to do with history length? unecessary imho.
        self.DQN = DQN_module(module_specs)
        self.device = self.DQN.device
        print_ts("DQN module has been initalized")

    def zip_module_specs(self):
        """
        """
        module_specs = {
            'BATCH_SIZE': self.batch_size,
            'GAMMA': self.gamma,
            'HISTORY_LENGTH': self.history_length,
            'SIZE_REPLAYBUFFER': self.size_replaybuffer,
            'OPTIM_LR': self.optim_learning_rate,
            'DIM_ACTIONS': self.dim_actions
            }
        return module_specs

	###########################################################################
    # Policy
    ###########################################################################

    def prepare_timestep(self, obs, reward, done, info):
        """
        Prepares the timestep:
        1. Incrementing counters
        2. Setting flags
        3. Retrieving information from the observation
        4. Calculating addtional information
        """
        # 1. Incrementing counters
        self.steps += 1  # from PYSC2 base class
        self.reward += reward  # from PYSC2 base class

        # Current episode
        self.timesteps += 1
        self.episode_reward_env += reward

        # 2. Setting flags
        # No flags

        # 3. Retrieving information from the observation
        self.state = np.array(obs[0], dtype=np.uint8)  # to reduce memory space
        self.first = obs[1]
        self.last = obs[2]
        # obs[3] unused
        # obs[4] unused
        self.available_actions = obs[5]

        # 4. Calculating addtional information
        # Setting up the index action space
        # self.available_Q_idx = []
        # for x in self.available_actions:
        #     self.available_Q_idx.append(PYSC2_to_Qvalueindex[x])
        # # print("Available Q indices are: " + str(self.available_Q_idx))

    def policy(self, obs, reward, done, info):
        """
        Choosing an action according to the agent mode.
        """
        # Set all variables at the start of a new timestep
        self.prepare_timestep(obs, reward, done, info)
        # Action selection for regular step
        if not self.last:
            if self.mode == 'supervised':
                # For the first n episodes learn on forced actions.
                self.action, self.action_idx = self.supervised_action()
            elif self.mode == 'learning':
                # Choose an action according to the policy
                self.action, self.action_idx = self.epsilon_greedy_action()
            elif self.mode == 'testing':
                self.action, self.action_idx = self.pick_action()
            else:
                print("no action selected")
                self.action, self.action_idx = 'no action', 'no action_idx'

        if self.last:  # End episode in last step
            self.action = 'reset'
            if self.mode != 'testing':
                self.update_target_network()
            self.reset()

        return self.action

    def pick_action(self):
        action_idx = self.DQN.pick_best(self.state)
        best_action = self.action_space[action_idx]
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
            chosen_action = self.action_space[action_idx]
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

    def supervised_action(self):
        """
        This method selects an action which will force the marine in the
        direction of the beacon. Further improvements are possible.
        """
        raise NotImplementedError

    # ##########################################################################
    # Evaluate a timestep
    # ##########################################################################

    def evaluate(self, obs, reward, done, info):
        """
        A generic wrapper, that contains all agent operations which are used
        after finishing a timestep.

        Retuns a dictionary with the following information:
            - Shaped reward per episode
            - Shaped reward cumulative
            - Mean loss per episode
            - epsilon progression per episode
        """

        if self.mode != 'testing':
            # Saving the episode data. Pushing the information onto the memory.
            self.store_transition(obs, reward)

            # Optimize the agent
            self.optimize()

        # collect reward, loss and epsilon information as dictionary
        agent_report = self.collect_report(obs, reward, done)

        return agent_report

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

        if self.get_memory_length() >= self.batch_size:
            self.list_loss_per_episode.append(self.loss.item())
        else:
            self.list_loss_per_episode.append(self.loss)

        if done:
            self.list_shaped_reward_cumulative.append(self.shaped_reward_cumulative)
            self.list_shaped_reward_per_episode.append(self.shaped_reward_per_episode)

            self.list_loss_mean.append(np.mean(self.list_loss_per_episode))

            # score observation per episode from pysc2 is appended
            self.list_pysc2_reward_per_episode.append(obs[6])

            # cumulative pysc2 reward
            self.pysc2_reward_cumulative += obs[6]
            self.list_pysc2_reward_cumulative.append(self.pysc2_reward_cumulative)

            # if self.episodes > self.supervised_episodes:
            self.list_epsilon_progression.append(self.epsilon)
            # else:
                # self.list_epsilon_progression.append(self.eps_start)

            dict_agent_report = {
                 "ShapedRewardPerEpisode": self.list_shaped_reward_per_episode,
                 "ShapedRewardCumulative": self.list_shaped_reward_cumulative,
                 "Pysc2RewardPerEpisode": self.list_pysc2_reward_per_episode,
                 "Pysc2RewardCumulative": self.list_pysc2_reward_cumulative,
                 "MeanLossPerEpisode": self.list_loss_mean,
                 "Epsilon": self.list_epsilon_progression,
                }

            print("Last epsilon: {}| Memory length: {}".format(
                    self.list_epsilon_progression[-1],
                    self.get_memory_length()))
            return dict_agent_report

    def store_transition(self, next_obs, reward):
        """
        Save the actual information in the history. As soon as there has been
        enough data, the experience is sampled from the replay buffer.
        TODO: Maybe using uint8 for storing into ReplayBuffer
        """
        # Don't store transition if first or last step
        if self.last:
            return

        self.reward = reward
        self.next_state = np.array(next_obs[0], dtype=np.uint8)

        self.DQN.memory.push([self.state],
                             [self.action_idx],
                             self.reward,
                             [self.next_state])

    # ##########################################################################
    # Mode switch
    # ##########################################################################

    def set_learning_mode(self):
        """
        Set the agent to learning mode. The agent will perform
        actions according to an epsilon-greedy policy.
        """
        print_ts("Setting agent to learning mode")
        self.mode = "learning"

    def set_supervised_mode(self):
        """
        Set the agent to supervised mode. The agent will perform
        actions which are generating valuable experience.
        """
        print_ts("Setting agent to supervised mode")
        self.mode = "supervised"

    def set_testing_mode(self):
        """
        Set the agent to testing mode. The agent will perform only
        exploiting actions without any randomness.
        """
        print_ts("Setting agent to testing mode")
        self.mode = "testing"

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
        if self.get_memory_length() >= self.batch_size:
            self.loss = self.DQN.optimize()

    def extract_skill(self):
        return self.DQN

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
        super(DQNBaseAgent, self).reset()
        self.timesteps = 0
        self.episode_reward_env = 0
        self.episode_reward_shaped = 0

        self.shaped_reward_per_episode = 0
        self.list_loss_per_episode = []

    # ##########################################################################
    # DQN module wrappers
    # ##########################################################################

    def save_model(self, exp_path, emergency=False):
        if emergency:
            print("KeyboardInterrupt detected, saving last model!")
            save_path = exp_path + "/model/emergency_model_{}.pt".format(self.episodes)
        else:
            save_path = exp_path + "/model/model_{}.pt".format(self.episodes)

        # Path((self.exp_path + "/model")).mkdir(parents=True, exist_ok=True)
        self.DQN.save(save_path)

    def load_model(self, load_path):
        self.DQN.load(load_path)

    def _load_emergency_model(self):
        load_path = self.exp_path + "/model/emergency_model.pt"
        self.DQN.load(load_path)

    #
    # def pick_action(self):
    #     """
    #     The agent picks the best available action. The pipeline acts as follows:
    #     1. Predict the Q-values for the whole action space
    #     2. Find the maximum of the available actions
    #     3. Return the action/action index of the full action space
    #     """
    #     # 1. Predict the Q-values for the whole action space
    #     action_q_values = self.DQN.predict_q_values(self.state)
    #     action_q_values_numpy = action_q_values.detach().cpu().numpy()
    #
    #     # 2. Find the maximum of the available actions
    #     print("Picking action from: " + str(self.available_actions))
    #     Q_avail_actions = action_q_values_numpy.T[self.available_Q_idx]
    #     # print("Action values for available actions: " + str(Q_avail_actions))
    #     Q_max_avail_action_idx = np.argmax(Q_avail_actions)
    #     # print("Action index of max action: " + str(action_idx))
    #
    #     # 3. Return the action/action index of the full action space
    #     action_idx = self.available_Q_idx[Q_max_avail_action_idx]
    #     # print("Best index of action space: "+str(best_action_idx_original))
    #     # PYSC2_action_idx = ALL_ACTIONS_PYSC2_IDX[best_action_idx_original]
    #     # print(PYSC2_action_idx)
    #     best_action = self.action_space[action_idx]
    #
    #     print("The best available action is : " + str(action_idx))
    #     chosen_action = best_action
    #     return chosen_action, action_idx
    #
    # def epsilon_greedy_action(self):
    #     """
    #     Chooses an action according to an epsilon-greedy policy
    #
    #     returns the chosen action id with x,y coordinates and the index of the
    #     action with respect to the SMART_ACTIONS constant. The additional index
    #     is used for the catergorical labeling of the actions as an
    #     "intermediate" solution for a restricted action space.
    #     """
    #     self.choice = self.epsilon_greedy()
    #     if self.choice == 'random':
    #         n_actions = len(self.available_actions)
    #         avail_action_idx = np.random.randint(n_actions)
    #         action_idx = self.available_actions[avail_action_idx]
    #         print("Chosen action idx: " + str(action_idx))
    #         action_idx = PYSC2_to_Qvalueindex[action_idx]
    #         chosen_action = self.action_space[action_idx]
    #     else:
    #         chosen_action, action_idx = self.pick_action()
    #     return chosen_action, action_idx


    # def pick_action(self):
    #     """
    #     The agent picks the best available action. The pipeline acts as follows:
    #     1. Predict the Q-values for the whole action space
    #     2. Find the maximum of the available actions
    #     3. Return the action/action index of the full action space
    #     """
    #     # 1. Predict the Q-values for the whole action space
    #     action_q_values = self.DQN.predict_q_values(self.state)
    #     action_q_values_numpy = action_q_values.detach().cpu().numpy()
    #
    #     # 2. Find the maximum of the available actions
    #     print("Picking action from: " + str(self.available_actions))
    #     Q_avail_actions = action_q_values_numpy.T[self.available_Q_idx]
    #     # print("Action values for available actions: " + str(Q_avail_actions))
    #     Q_max_avail_action_idx = np.argmax(Q_avail_actions)
    #     # print("Action index of max action: " + str(action_idx))
    #
    #     # 3. Return the action/action index of the full action space
    #     action_idx = self.available_Q_idx[Q_max_avail_action_idx]
    #     # print("Best index of action space: "+str(best_action_idx_original))
    #     # PYSC2_action_idx = ALL_ACTIONS_PYSC2_IDX[best_action_idx_original]
    #     # print(PYSC2_action_idx)
    #     best_action = self.action_space[action_idx]
    #
    #     print("The best available action is : " + str(action_idx))
    #     chosen_action = best_action
    #     return chosen_action, action_idx
    #
    # def epsilon_greedy_action(self):
    #     """
    #     Chooses an action according to an epsilon-greedy policy
    #
    #     returns the chosen action id with x,y coordinates and the index of the
    #     action with respect to the SMART_ACTIONS constant. The additional index
    #     is used for the catergorical labeling of the actions as an
    #     "intermediate" solution for a restricted action space.
    #     """
    #     self.choice = self.epsilon_greedy()
    #     if self.choice == 'random':
    #         n_actions = len(self.available_actions)
    #         avail_action_idx = np.random.randint(n_actions)
    #         action_idx = self.available_actions[avail_action_idx]
    #         print("Chosen action idx: " + str(action_idx))
    #         action_idx = PYSC2_to_Qvalueindex[action_idx]
    #         chosen_action = self.action_space[action_idx]
    #     else:
    #         chosen_action, action_idx = self.pick_action()
    #     return chosen_action, action_idx
