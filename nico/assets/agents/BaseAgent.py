import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from pysc2.agents import base_agent
from pysc2.lib import actions, features
from collections import namedtuple
import math

from AtariNet import DQN
from assets.smart_actions import SMART_ACTIONS_SIMPLE_NAVIGATION as SMART_ACTIONS
from assets.memory.ReplayBuffer import ReplayBuffer
from assets.memory.History import History
from assets.helperFunctions.flagHandling import set_flag_every, set_flag_from
from assets.helperFunctions.timestamps import print_timestamp as print_ts

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)  # only for terminal output visualization


class BaseAgent(base_agent.BaseAgent):
    """
    This is a simple agent that uses an AtariNet action value approximator.
    """
    # ##########################################################################
    # Initializing the agent
    # ##########################################################################

    def __init__(self, agent_file):
        """

        """
        super(BaseAgent, self).__init__()
        self.unzip_hyperparameter_file(agent_file)
        self.steps_done = 0  # total_timesteps
        self.timesteps = 0  # timesteps in the current episode
        self.update_cnt = 0  # Target network update_counter
        self.choice = None  # Choice of epsilon greedy
        self.loss = 0  # Action loss

        self.memory = ReplayBuffer(self.size_replaybuffer)
        self.hist = History(self.history_length)
        self.net, self.target_net, self.optimizer = self._build_model()
        self.print_architecture()

    def unzip_hyperparameter_file(self, agent_file):
        """
        Unzipping the hyperparameter files and writing them into member variables.
        """
        self.screen_dim = agent_file['SCREEN_DIM']
        self.minimap_dim = agent_file['MINIMAP_DIM']
        self.x_map_dim = 84  # magic number
        self.y_map_dim = 64  # magic number
        self.action_dim = len(SMART_ACTIONS)
        self.map_dimensions = (self.x_map_dim, self.y_map_dim)
        self.gamma = agent_file['GAMMA']
        self.optim_learning_rate = agent_file['OPTIM_LR']
        self.batch_size = agent_file['BATCH_SIZE']
        self.target_update_period = agent_file['TARGET_UPDATE_PERIOD']
        self.history_length = agent_file['HIST_LENGTH']
        self.size_replaybuffer = agent_file['REPLAY_SIZE']
        self.device = agent_file['DEVICE']
        self.silentmode = agent_file['SILENTMODE']
        self.supervised_episodes = agent_file['SUPERVISED_EPISODES']

        epsilon_file = agent_file['EPSILON_FILE']
        self.epsilon = epsilon_file['EPSILON']
        self.eps_start = epsilon_file['EPS_START']
        self.eps_end = epsilon_file['EPS_END']
        self.eps_decay = epsilon_file['EPS_DECAY']

    def setup_interface(self):
        """
        Setting up agent interface for the environment.
        """
        self.agent_interface = features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=self.screen_dim, minimap=self.minimap_dim),
                            use_feature_units=True)
        return self.agent_interface

    def _build_model(self):
        """
        Initializing 2 networks and an Adam optimizer.
        """
        net = DQN(self.history_length).to(self.device)
        target_net = DQN(self.history_length).to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.optim_learning_rate)
        return net, target_net, optimizer

    def print_architecture(self):
        print_ts("Network: \n{}".format(self.net))
        print_ts("Optimizer: \n{}".format(self.optimizer))
        print_ts("Target Network: \n{}".format(self.target_net))

    # ##########################################################################
    # Action Selection
    # ##########################################################################

    def prepare_timestep(self, obs, last_score):
        """
        Initializing variables and flags.
        The obs container will be written to member variables.
        This shall keep the memory cleaner during the actual step.
        """
        self.timesteps += 1
        self.steps += 1  # From PYSC2 base class

        self.reward += obs.reward  # total_reward (PYSC2 convention)
        self.step_type = obs.step_type  # Important to check for last step
        self.available_actions = obs.observation.available_actions

        # Calculate additional information for reward shaping
        self.feature_screen = obs.observation.feature_screen.player_relative
        self.feature_screen2 = obs.observation.feature_screen.selected
        self.beacon_center, self.marine_center, self.distance = self.calculate_distance(self.feature_screen, self.feature_screen2)

        self.last_score = last_score

        self.state = torch.tensor([self.feature_screen], dtype=torch.float, device=self.device)
        self.history_tensor = self.state
        self.state_history_tensor = self.state.unsqueeze(1)

    def calculate_distance(self, screen_player_relative, screen_selected):
        """
        Calculates the euclidean distance between beacon and marine.
        Using feature_screen.selected since marine vanishes behind beacon when
        using feature_screen.player_relative
        """
        marine_center = np.mean(self._xy_locs(screen_selected == 1), axis=0).round()
        beacon_center = np.mean(self._xy_locs(screen_player_relative == 3), axis=0).round()
        if isinstance(marine_center, float):
            marine_center = beacon_center
        distance = math.hypot(beacon_center[0] - marine_center[0],
                              beacon_center[1] - marine_center[1])

        return beacon_center, marine_center, distance

    def step(self, agent_mode):
        """
        Choosing an action
        """
        # For the first n episodes learn on forced actions.
        if self.episodes < self.supervised_episodes:
            self.action, self.action_idx = self.supervised_action()
        else:
            # Choose an action according to the policy
            self.action, self.action_idx = self.choose_action(agent_mode)
        return self.action

    def supervised_action(self):
        """
        This method selects an action which will force the marine in the
        direction of the beacon.
        Further improvements are possible.
        """
        relative_vector = self.marine_center - self.beacon_center
        action_choice = np.random.choice([True, False], p=[0.5, 0.5])

        right_to_beacon = relative_vector[0] > 0.
        left_to_beacon = relative_vector[0] < 0.
        vertically_aligned = relative_vector[0] == 0.

        below_beacon = relative_vector[1] > 0.
        above_beacon = relative_vector[1] < 0.
        horizontally_aligned = relative_vector[1] == 0.

        # print("Test 1: {}, {}".format(horizontally_aligned, right_to_beacon))
        if (horizontally_aligned and vertically_aligned):
            chosen_action = self.check_action(SMART_ACTIONS[self.action_idx])
            return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)

        # print("Test 1: {}, {}".format(horizontally_aligned, right_to_beacon))
        if (horizontally_aligned and right_to_beacon):
            chosen_action = self.check_action('left')
            self.action_idx = 0
            return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)

        # print("Test 2: {}, {}".format(below_beacon, vertically_aligned))
        if (below_beacon and vertically_aligned):
            chosen_action = self.check_action('up')
            self.action_idx = 1
            return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)

        # print("Test 3: {}, {}".format(horizontally_aligned, left_to_beacon))
        if (horizontally_aligned and left_to_beacon):
            chosen_action = self.check_action('right')
            self.action_idx = 2
            return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)

        # print("Test 4: {}, {}".format(above_beacon, vertically_aligned))
        if (above_beacon and vertically_aligned):
            chosen_action = self.check_action('down')
            self.action_idx = 3
            return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)

        # print("Test 5: {}, {}".format(above_beacon, left_to_beacon))
        if above_beacon and left_to_beacon:
            # print("Test 5.1")
            if action_choice:
                chosen_action = self.check_action('right')
                self.action_idx = 2
                return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)
            # print("Test 5.2")
            if not action_choice:
                chosen_action = self.check_action('down')
                self.action_idx = 3
                return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)
        # print("Test 6: {}, {}".format(above_beacon, right_to_beacon))
        if (above_beacon and right_to_beacon):
            # print("Test 6.1")
            if action_choice:
                chosen_action = self.check_action('left')
                self.action_idx = 0
                return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)
            # print("Test 6.2")
            if not action_choice:
                chosen_action = self.check_action('down')
                self.action_idx = 3
                return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)
        # print("Test 7: {}, {}".format(below_beacon, right_to_beacon))
        if (below_beacon and right_to_beacon):
            # print("Test 7.1")
            if action_choice:
                chosen_action = self.check_action('left')
                self.action_idx = 0
                return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)
            # print("Test 7.2")
            if not action_choice:
                chosen_action = self.check_action('up')
                self.action_idx = 1
                return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)
        # print("Test 8: {}, {}".format(below_beacon, left_to_beacon))
        if (below_beacon and left_to_beacon):
            # print("Test 8.1")
            if action_choice:
                chosen_action = self.check_action('right')
                self.action_idx = 2
                return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)
            # print("Test 8.2")
            if not action_choice:
                chosen_action = self.check_action('up')
                self.action_idx = 1
                return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)

    def choose_action(self, agent_mode):
        """
        chooses an action according to the current policy
        returns the chosen action id with x,y coordinates and the index of the action
        with respect to the SMART_ACTIONS constant. The additional index is used for
        the catergorical labeling of the actions as an "intermediate" solution for
        a restricted action space.
        """
        self.choice = self.epsilon_greedy()

        if self.choice == 'random' and agent_mode == 'learn':
            self.action_idx = np.random.randint(self.action_dim)
            chosen_action = self.check_action(SMART_ACTIONS[self.action_idx])
        else:
            # Q-values berechnen
            with torch.no_grad():
                action_q_values = self.net(self.state_history_tensor)

            # Beste Action bestimmen
            best_action_numpy = action_q_values.detach().numpy()
            self.action_idx = np.argmax(best_action_numpy)
            best_action = SMART_ACTIONS[self.action_idx]

            chosen_action = self.check_action(best_action)
        # square brackets around chosen_action needed for internal pysc2 state machine
        return [chosen_action], torch.tensor([self.action_idx], dtype=torch.long, device=self.device)

    def check_action(self, action):
        """
        gets action from id list and passes valid args
        TODO: make this more intuitive and general
        """
        if self.can_do(actions.FUNCTIONS.Move_screen.id):
            if action is 'left':
                if not (self.marine_center[0] == 0):
                    return actions.FUNCTIONS.Move_screen("now", self.marine_center + (-2, 0))
            if action is 'up':
                if not (self.marine_center[1] == 0):
                    return actions.FUNCTIONS.Move_screen("now", self.marine_center + (0, -2))
            if action is 'right':
                if not (self.marine_center[0] == 83):
                    return actions.FUNCTIONS.Move_screen("now", self.marine_center + (2, 0))
            if action is 'down':
                if not (self.marine_center[1] == 63):
                    return actions.FUNCTIONS.Move_screen("now", self.marine_center + (0, 2))
        else:
            return actions.FUNCTIONS.no_op()

    def can_do(self, action):
        """
        shortcut for checking if action is available at the moment
        """
        return action in self.available_actions

    def epsilon_greedy(self):
        """
        returns a string in order to determine if the next action choice is
        going to be random or according to an decaying epsilon greeedy policy
        """
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        return np.random.choice(['random', 'greedy'], p=[self.epsilon, 1-self.epsilon])

    def _xy_locs(self, mask):
        """
        Mask should be a set of bools from comparison with a feature layer.
        """
        y, x = mask.nonzero()
        return list(zip(x, y))

    # ##########################################################################
    # Store transition in Replay Buffer
    # ##########################################################################

    def store_transition(self, obs):
        """
        Save the actual information in the history.
        As soon as there has been enough data, the experience is sampled from the replay buffer.
        """
        self.env_reward = obs.reward
        self.feature_screen_next = obs.observation.feature_screen.player_relative
        self.feature_screen_next2 = obs.observation.feature_screen.selected
        self.reward_shaping()

        # collect transition data
        reward = torch.tensor([self.reward_shaped], device=self.device, dtype=torch.float)
        step_type = torch.tensor([obs.step_type], device=self.device, dtype=torch.int)
        next_state = torch.tensor([self.feature_screen_next], dtype=torch.float, device=self.device)

        # push next state on next state history stack
        next_state_history_tensor = next_state

        # save transition tuple to the memory buffer
        self.memory.push(self.state_history_tensor, self.action_idx, reward, next_state_history_tensor, step_type)

    def reward_shaping(self):
        """
        A new reward will be calculated based on the distance covered by the agent in the last step.
        """
        self.beacon_center_next, self.marine_center_next, self.distance_next = self.calculate_distance(self.feature_screen_next, self.feature_screen_next2)
        self.reward_shaped = self.distance - self.distance_next
        if self.distance == 0.0:
            self.reward_shaped = 100
        # self.reward_combined = self.reward + self.reward_shaped


    # ##########################################################################
    # Print status information of timestep
    # ##########################################################################

    def print_status(self):
        """
        function helper for status printing
        """
        if self.episodes < self.supervised_episodes:
            q_max, q_min, q_mean, q_var, q_span, q_argmax = self.q_value_analysis(self.state_q_values)
            td_max, td_min, td_mean, td_var, td_span, td_argmax = self.q_value_analysis(self.td_target.unsqueeze(1))
            # print("environment reward: {:.2f}, step penalty: {:.2f}, reward total: {:.2f}".format(obs.reward, self.reward_shaped, self.reward_combined))
            # print("Q_VALUES: {}".format(self.state_q_values))
            # print("TD_TARGET: {}".format(self.td_target.unsqueeze(1)))
            print_ts("action: {}, idx: {}, Smart action: {}".format(self.action, self.action_idx, SMART_ACTIONS[self.action_idx]))
            print_ts("Q_VALUES: max: {:.3f}, min: {:.3f}, span: {:.3f}, mean: {:.3f}, var: {:.6f}, argmax: {}".format(q_max, q_min, q_span, q_mean, q_var, q_argmax))
            print_ts("TD_TARGET: max: {:.3f}, min: {:.3f}, span: {:.3f}, mean: {:.3f}, var: {:.6f}, argmax: {}".format(td_max, td_min, td_span, td_mean, td_var, td_argmax))
            print_ts("MEMORY SIZE: {}".format(len(self.memory)))
            print_ts("Epsilon: {:.2f}\t| choice: {}".format(self.epsilon, self.choice))
            print_ts("Episode {}\t| Step {}\t| Total Steps: {}".format(self.episodes, self.timesteps, self.steps))
            print_ts("Chosen action: {}".format(self.action))
            # print_ts("chosen coordinates [x,y]: {}".format((self.x_coord.item(), self.y_coord.item())))
            print_ts("Beacon center location [x,y]: {}".format(self.beacon_center))
            print_ts("Marine center location [x,y]: {}".format(self.marine_center))
            print_ts("Distance: {}, delta distance: {}".format(self.distance, self.reward_shaped))
            print_ts("Current Episode Score: {}\t| Total Score: {}".format(self.last_score, self.reward))
            print_ts("Environment reward in timestep: {}".format(self.env_reward))
            print_ts("Action Loss: {:.5f}".format(self.loss))
            print_ts("----------------------------------------------------------------")
        else:
            print(self.feature_screen)

    # ##########################################################################
    # Optimizing the network
    # ##########################################################################

    def optimize(self):
        """
        optimizes the model. currently only trains the actions
        # TODO: extend Q-Update function to the x and y coordinates
        """

        # Sample batch from experience replay buffer
        batch = self.sample_batch()

        # Unzip batch data
        self.unzip_batch(batch)

        # Calculate Q-values
        self.calculate_q_values()

        # calculate td targets of the actions, x&y coordinates
        self.td_target = self.calculate_td_target(self.next_state_q_values_max)

        # Compute the loss
        self.compute_loss()

        # optimize model
        self.optimize_model()

    def sample_batch(self):
        """
        Sample from batch.
        """
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'step_type'))
        transitions = self.memory.sample(self.batch_size)
        return Transition(*zip(*transitions))

    def unzip_batch(self, batch):
        """
        Get the batches from the transition tuple
        """
        self.state_batch = torch.cat(batch.state)
        self.action_batch = torch.cat(batch.action).unsqueeze(1)
        self.reward_batch = torch.cat(batch.reward)
        self.step_type_batch = torch.cat(batch.step_type)
        self.next_state_batch = torch.cat(batch.next_state).unsqueeze(1)

    def calculate_q_values(self):
        """
        """
        # forward pass
        state_q_values = self.net(self.state_batch)

        # Debugging
        # _, _, _, _, _, max_action = self.q_value_analysis_on_batch(state_q_values)
        # print(self.timesteps, max_action)
        # print(self.action_batch.detach().numpy())

        # gather action values with respect to the chosen action
        self.state_q_values = state_q_values.gather(1, self.action_batch)
        # compute action values of the next state over all actions and take the max
        next_state_q_values = self.target_net(self.next_state_batch)
        # print("next_state_q_values")
        # print(next_state_q_values)
        self.next_state_q_values_max = next_state_q_values.max(1)[0].detach()
        # print("next_state_q_values_max")
        # print(self.next_state_q_values_max)

    def calculate_td_target(self, q_values_best_next):
        """
        Calculating the TD-target for given q-values.
        """
        # td_target = torch.tensor((q_values * self.gamma) + self.reward_batch, dtype=torch.float, device=self.device, requires_grad=True)
        td_target = (q_values_best_next * self.gamma) + self.reward_batch
        # print(q_values_best_next)
        # print(self.reward_batch)
        # print("td_target")
        # print(td_target)
        # td_target[np.where(self.step_type_batch == 2)] = self.reward_batch[np.where(self.step_type_batch == 2)]
        return td_target

    def q_value_analysis_on_batch(self, q_value_tensor):
        """
        """
        q_max = q_value_tensor.detach().numpy().max(axis=1).mean()
        q_min = q_value_tensor.detach().numpy().min(axis=1).mean()
        q_mean = q_value_tensor.detach().numpy().mean(axis=1).mean()
        q_var = q_value_tensor.detach().numpy().var(axis=1).mean()
        q_argmax = q_value_tensor.detach().numpy().argmax(axis=1)
        q_span = q_max - q_min
        return q_max, q_min, q_mean, q_var, q_span, q_argmax

    def q_value_analysis(self, q_value_tensor):
        """
        """
        q_max = q_value_tensor.detach().numpy().max()
        q_min = q_value_tensor.detach().numpy().min()
        q_mean = q_value_tensor.detach().numpy().mean()
        q_var = q_value_tensor.detach().numpy().var()
        q_argmax = q_value_tensor.detach().numpy().argmax()
        q_span = q_max - q_min
        return q_max, q_min, q_mean, q_var, q_span, q_argmax

    def compute_loss(self):
        """
        Calculating the loss between TD-target Q-values.
        """
        self.loss = F.mse_loss(self.state_q_values, self.td_target.unsqueeze(1))  # compute MSE loss

    def optimize_model(self):
        """
        This step is used to update the parameter weights:
        loss.backwards propagates the loss gradients for all net weights backwards.
        optimizer.step updates all net weights with the assigned gradients.
        optimizer.zero_grad sets the gradients to 0 for the next update cycle.
        """

        # for param in list(self.net.parameters()):
        #     print(param.grad.data)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    # ##########################################################################
    # Ending Episode
    # ##########################################################################

    def update_target_network(self):
        """
        Transferring the estimator weights to the target weights
        """
        if self.episodes % self.target_update_period == 0:
            print_ts("About to update")
            self.update_target_net(self.net, self.target_net)
        self.reset()

    def reset(self):
        """
        Resetting the agent --> More explanation
        """
        super(BaseAgent, self).reset()
        self.timesteps = 0

    def update_target_net(self, src_net, target_net):
        """
        updates weights of the target network, i.e. copies model weights to it
        """
        target_net.load_state_dict(src_net.state_dict())
        self.update_cnt += 1

    def get_net_weights(self):
        """
        Get the weights of the neural network
        """
        return self.net.state_dict()

    def set_net_weights(self, weights):
        """
        Set the weights of the neural network
        """
        self.net.load_state_dict

    def get_target_net_weights(self):
        return self.target_net.state_dict()

    def set_target_net_weights(self, weights):
        self.target_net.load_state_dict

if __name__ == '__main__':
    BaseAgent = BaseAgent()

    # ##########################################################################
    # Code dumpster
    # ##########################################################################
    # def set_episode_flags(self, ep):
    #     """
    #     This private method sets and resets the flags:
    #       render: indicates if progess has to be rendered
    #       store: indicates if progress has to be stored
    #       save: indicates if weight file has to be saved
    #       test: indicates if testrun has to be started
    #       update: indicates when weights are starting to be updated
    #     """
    #     self.render = set_flag_every(self.show_progress, ep)
    #     self.store = set_flag_every(self.store_progress, ep)
    #     self.save = set_flag_every(self.auto_saver, ep)
    #     self.test = set_flag_every(self.test_each, ep)
    #     self.update = set_flag_from(10, ep)  # magic number
