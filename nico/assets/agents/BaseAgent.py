import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from pysc2.agents import base_agent
from pysc2.lib import actions, features
from collections import namedtuple
import math

from AtariNet import DQN
from assets.smart_actions import SMART_ACTIONS_MOVE2BEACON_SIMPLE as SMART_ACTIONS
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
    # Initializing the agent_file
    # ##########################################################################

    def __init__(self, agent_file):
        super(BaseAgent, self).__init__()
        self.unzip_hyperparameter_file(agent_file)
        self.steps_done = 0
        self.timesteps = 0
        self.update_cnt = 0
        self.choice = None
        self.loss = 0
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
        self.map_dimensions = (self.x_map_dim, self.y_map_dim)
        self.gamma = agent_file['GAMMA']
        self.optim_learning_rate = agent_file['OPTIM_LR']
        self.batch_size = agent_file['BATCH_SIZE']
        self.target_update_period = agent_file['TARGET_UPDATE_PERIOD']
        self.history_length = agent_file['HIST_LENGTH']
        self.size_replaybuffer = agent_file['REPLAY_SIZE']
        self.device = agent_file['DEVICE']
        self.silentmode = agent_file['SILENTMODE']

        epsilon_file = agent_file['EPSILON_FILE']
        self.epsilon = epsilon_file['EPSILON']
        self.eps_start = epsilon_file['EPS_START']
        self.eps_end = epsilon_file['EPS_END']
        self.eps_decay = epsilon_file['EPS_DECAY']

    def setup_interface(self):
        """
        Setting up agent interface for the environment.
        """
        # Maybe if-else to use RGB for game visualization
        self.agent_interface = features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=self.screen_dim, minimap=self.minimap_dim),
                            # rgb_dimensions = features.Dimensions(screen=SCREEN_DIM,
                            #                     minimap=MINIMAP_DIM),
                            # action_space = actions.ActionSpace.RGB,
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
    # Performing a step
    # ##########################################################################

    def step(self, obs, last_score, agent_mode):
        """
        Choosing an action
        """
        # Set all variables at the start of a new timestep
        self.initializing_timestep(obs, last_score)

        # Choose an action according to the policy
        self.action, self.action_idx, self.x_coord, self.y_coord = self.choose_action(self.available_actions, agent_mode)

        # Saving the episode data. Pushing the information onto the memory.
        if agent_mode is 'learn':
            self.save_data(obs)

            if len(self.memory) >= self.batch_size: #if.. return more performant (keeps stack cleaner)
                self.loss = self.optimize()

                # Print actual status information
                if not self.silentmode:
                    self.print_status()
        return self.action

    def initializing_timestep(self, obs, last_score):
        """
        Initializing variables and flags
        """
        self.reward += obs.reward
        self.step_type = obs.step_type
        self.available_actions = obs.observation.available_actions
        self.timesteps += 1
        self.steps += 1
        # self.actual_reward = obs.reward
        self.last_score = last_score
        self.feature_screen = obs.observation.feature_screen.player_relative
        self.beacon_center = np.mean(self._xy_locs(self.feature_screen == 3), axis=0).round()
        self.player_center = np.mean(self._xy_locs(self.feature_screen == 1), axis=0).round()
        self.state = torch.tensor([self.feature_screen], dtype=torch.float, device=self.device, requires_grad = True)
        # self.history_tensor = self.hist.stack(self.state)  # stack state on history
        self.history_tensor = self.state
        self.state_history_tensor = self.state
        self.state_history_tensor = self.state.unsqueeze(1)

    def choose_action(self, available_actions, agent_mode):
        """
        chooses an action according to the current policy
        returns the chosen action id with x,y coordinates and the index of the action
        with respect to the SMART_ACTIONS constant. The additional index is used for
        the catergorical labeling of the actions as an "intermediate" solution for
        a restricted action space.
        """
        self.choice = self.epsilon_greedy()
        if self.choice == 'random' and agent_mode == 'learn':
            action_idx = np.random.randint(len(SMART_ACTIONS))  # not clean
            x_coord = np.random.randint(self.x_map_dim)
            y_coord = np.random.randint(self.y_map_dim)
            chosen_action = self.get_action(available_actions, SMART_ACTIONS[action_idx], x_coord, y_coord)
        else:
            with torch.no_grad():
                action_q_values, x_coord_q_values, y_coord_q_values = self.net(self.state_history_tensor)
            action_idx = 0
            best_action = SMART_ACTIONS[action_idx]
            x_numpy = x_coord_q_values.detach().numpy()
            y_numpy = y_coord_q_values.detach().numpy()
            x_coord = np.argmax(x_numpy)
            y_coord = np.argmax(y_numpy)
            chosen_action = self.get_action(available_actions, best_action, x_coord, y_coord)
        # square brackets arounc chosen_action needed for internal pysc2 state machine
        return [chosen_action], torch.tensor([action_idx], dtype=torch.long, device=self.device), torch.tensor([x_coord], dtype=torch.long, device=self.device), torch.tensor([y_coord], dtype=torch.long, device=self.device)

    def get_action(self, available_actions, action, x, y):
        """
        gets action from id list and passes valid args
        TODO: make this more intuitive and general
        """
        if self.can_do(available_actions, action):
            if action == actions.FUNCTIONS.Move_screen.id:
                return actions.FUNCTIONS.Move_screen("now", (x, y))
            # if action == actions.FUNCTIONS.select_army.id:
            #     return actions.FUNCTIONS.select_army("select")
            if action == actions.FUNCTIONS.no_op.id:
                return actions.FUNCTIONS.no_op()
        else:
            return actions.FUNCTIONS.no_op()

    def can_do(self, available_actions, action):
        """
        shortcut for checking if action is available at the moment
        """
        return action in available_actions

    def epsilon_greedy(self):
        """
        returns a string in order to determine if the next action choice is
        going to be random or according to an decaying epsilon greeedy policy
        """
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        return np.random.choice(['random', 'greedy'], p=[self.epsilon, 1-self.epsilon])

    def set_episode_flags(self, ep):
        """
        This private method sets and resets the flags:
          render: indicates if progess has to be rendered
          store: indicates if progress has to be stored
          save: indicates if weight file has to be saved
          test: indicates if testrun has to be started
          update: indicates when weights are starting to be updated
        """
        self.render = set_flag_every(self.show_progress, ep)
        self.store = set_flag_every(self.store_progress, ep)
        self.save = set_flag_every(self.auto_saver, ep)
        self.test = set_flag_every(self.test_each, ep)
        self.update = set_flag_from(10, ep)  # magic number

    def end_episode(self, agent_mode):
        """
        """
        if agent_mode is 'learn':
            print_ts("About to update")
            if self.episodes % self.target_update_period == 0 and self.episodes != 0:
                self.update_target_net(self.net, self.target_net)
            self.reset()

    def _xy_locs(self, mask):
        """
        Mask should be a set of bools from comparison with a feature layer.
        """
        y, x = mask.nonzero()
        return list(zip(x, y))

    def sample_batch(self):
        """
        Sample from batch.
        """
        Transition = namedtuple('Transition', ('state', 'action', 'x_coord', 'y_coord', 'reward', 'next_state', 'step_type'))
        transitions = self.memory.sample(self.batch_size)
        return Transition(*zip(*transitions))

    def reset(self):
        """
        Resetting the agent --> More explanation
        """
        super(BaseAgent, self).reset()
        self.timesteps = 0
        self.update_cnt = 0

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

    def save_data(self, obs):
        """
        Save the actual information in the history.
        As soon as there has been enough data, the experience is sampled from the replay buffer.
        """
        # collect transition data
        reward = torch.tensor([obs.reward], device=self.device, dtype=torch.float)
        step_type = torch.tensor([obs.step_type], device=self.device, dtype=torch.int)
        next_state = torch.tensor([obs.observation.feature_screen.player_relative], dtype=torch.float, device=self.device, requires_grad = True)

        # push next state on next state history stack
        next_state_history_tensor = next_state

        # save transition tuple to the memory buffer
        self.memory.push(self.state_history_tensor, self.action_idx, self.x_coord, self.y_coord, reward, next_state_history_tensor, step_type)

    def print_status(self):
        """
        function helper for status printing
        """
        print_ts("Epsilon: {:.2f}\t| choice: {}".format(self.epsilon, self.choice))
        print_ts("Episode {}\t| Step {}\t| Total Steps: {}".format(self.episodes, self.timesteps, self.steps))
        print_ts("Chosen action: {}".format(self.action))
        print_ts("chosen coordinates [x,y]: {}".format((self.x_coord.item(), self.y_coord.item())))
        print_ts("Beacon center location [x,y]: {}".format(self.beacon_center))
        print_ts("Current Episode Score: {}\t| Total Score: {}".format(self.last_score, self.reward))
        print_ts("Action Loss: {:.5f}".format(self.loss))
        print_ts("----------------------------------------------------------------")

    # ##########################################################################
    # optimizing the network
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
        self.td_target_actions = self.calculate_td_target(self.next_state_q_values_max)
        self.td_target_x_coord = self.calculate_td_target(self.next_x_q_values_max)
        self.td_target_y_coord = self.calculate_td_target(self.next_y_q_values_max)

        # Compute the loss
        self.compute_loss()

        # optimize model
        self.optimize_model()

        return self.loss.item()

    def unzip_batch(self, batch):
        """
        Get the batches from the transition tuple
        """
        self.state_batch = torch.cat(batch.state)
        self.action_batch = torch.cat(batch.action).unsqueeze(1)
        self.x_coord_batch = torch.cat(batch.x_coord).unsqueeze(1)
        self.y_coord_batch = torch.cat(batch.y_coord).unsqueeze(1)
        self.reward_batch = torch.cat(batch.reward)
        self.step_type_batch = torch.cat(batch.step_type)
        self.next_state_batch = torch.cat(batch.next_state).unsqueeze(1)

    def calculate_q_values(self):
        """
        """
        # forward pass
        state_q_values, x_q_values, y_q_values = self.net(self.state_batch)

        # gather action values with respect to the chosen action
        self.state_q_values = state_q_values.gather(1, self.action_batch)
        self.x_q_values = x_q_values.gather(1, self.x_coord_batch)
        self.y_q_values = y_q_values.gather(1, self.y_coord_batch)


        # compute action values of the next state over all actions and take the max
        next_state_q_values, next_x_q_values, next_y_q_values = self.target_net(self.next_state_batch)

        self.next_state_q_values_max = next_state_q_values.max(1)[0].detach()
        self.next_x_q_values_max = next_x_q_values.max(1)[0].detach()
        self.next_y_q_values_max = next_y_q_values.max(1)[0].detach()

    def calculate_td_target(self, q_values):
        """
        Calculating the TD-target for given q-values.
        """
        # td_target = torch.tensor((q_values * self.gamma) + self.reward_batch, dtype=torch.float, device=self.device, requires_grad=True)
        td_target = (q_values * self.gamma) + self.reward_batch
        # td_target[np.where(self.step_type_batch == 2)] = self.reward_batch[np.where(self.step_type_batch == 2)]
        return td_target

    def q_value_analysis(self, q_value_tensor):
        """
        """
        q_max = q_value_tensor.max().detach().numpy()
        q_min = q_value_tensor.min().detach().numpy()
        q_mean = q_value_tensor.mean().detach().numpy()
        q_var = q_value_tensor.var().detach().numpy()
        q_span = q_max - q_min
        return q_max, q_min, q_mean, q_var, q_span

    def compute_loss(self):
        """
        Calculating the TD-target for given q-values.
        """
        q_values_cat = torch.cat((self.state_q_values, self.x_q_values, self.y_q_values))
        td_target_cat = torch.cat((self.td_target_actions, self.td_target_x_coord, self.td_target_y_coord))


        # print(q_values_cat)
        # print(td_target_cat)
        # print("Q_VALUES {}".format(self.x_q_values))
        q_x_max, q_x_min, q_x_mean, q_x_var, q_x_span = self.q_value_analysis(self.x_q_values)
        q_y_max, q_y_min, q_y_mean, q_y_var, q_y_span = self.q_value_analysis(self.y_q_values)
        # print("----------------------------------------------------------------------------")
        # print("Q_VALUES X max: {:.3f}, min: {:.3f}, span: {:.3f}, mean: {:.3f}, var: {:.6f}".format(q_x_max, q_x_min, q_x_span, q_x_mean, q_x_var))
        # print("Q_VALUES Y max: {:.3f}, min: {:.3f}, span: {:.3f}, mean: {:.3f}, var: {:.6f}".format(q_y_max, q_y_min, q_y_span, q_y_mean, q_y_var))

        # compute MSE loss
        # self.loss = F.mse_loss(q_values_cat, td_target_cat.unsqueeze(1))
        # self.loss = F.mse_loss(q_values_cat, td_target_cat.unsqueeze(1))
        # loss1 = F.mse_loss(self.state_q_values, self.td_target_actions.unsqueeze(1))
        loss2 = F.mse_loss(self.x_q_values, self.td_target_x_coord.unsqueeze(1))
        loss3 = F.mse_loss(self.y_q_values, self.td_target_y_coord.unsqueeze(1))

        self.loss = loss2 + loss3

        # print("TOTAL_LOSS: {:.6f}     x_loss: {:.6f}, y_loss: {:.6f}".format(self.loss, loss2, loss3))
        # print("----------------------------------------------------------------------------")
        # print("Q_VALUES {}".format(q_values_cat))
        # print("TD targets {}".format(td_target_cat.unsqueeze(1)))
        # print("loss: {}".format(self.loss))

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

if __name__ == '__main__':
    BaseAgent = BaseAgent()

    # ##########################################################################
    # Code dumpster
    # ##########################################################################

    # def optimize(self):
    #     """
    #     optimizes the model. currently only trains the actions
    #     # TODO: extend Q-Update function to the x and y coordinates
    #     """
    #
    #     # Sample batch from experience replay buffer
    #     batch = self.sample_batch()
    #
    #     # Unzip batch data
    #     self.unzip_batch(batch)
    #
    #     # print(self.state_batch)
    #     # print(self.action_batch)
    #     # print(self.x_coord_batch)
    #     # print(self.y_coord_batch)
    #
    #
    #     # Calculate Q-values
    #     self.calculate_q_values()
    #
    #     # print("Q-values for the batch")
    #     # print(self.state_q_values)
    #     # print(self.x_q_values)
    #     # print(self.y_q_values)
    #     #
    #     # print("max values for next step")
    #     # print(self.next_state_q_values_max)
    #     # print(self.next_x_q_values_max)
    #     # print(self.next_y_q_values_max)
    #
    #     # calculate td targets of the actions, x&y coordinates
    #     self.td_target_actions = self.calculate_td_target(self.next_state_q_values_max)
    #     self.td_target_x_coord = self.calculate_td_target(self.next_x_q_values_max)
    #     self.td_target_y_coord = self.calculate_td_target(self.next_y_q_values_max)
    #
    #     # Compute the loss
    #     self.compute_loss()
    #
    #     # optimize model
    #     self.optimize_model()
    #
    #     return self.loss.item()

    # def unit_type_is_selected(self, obs, unit_type):
    #     """
    #     Check if certain unit type is selected at moment of call, supports single
    #     select and multi_select
    #     """
    #     if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
    #         return True
    #     if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
    #         return True
    #     return False
    #
    # def get_unit_by_type(self, obs, unit_type):
    #     """
    #     select units by type, i.e. CTRL + LEFT_CLICK (on screen)
    #     """
    #     return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    # def choose_action(self, available_actions, agent_mode):
    #     """
    #     chooses an action according to the current policy
    #     returns the chosen action id with x,y coordinates and the index of the action
    #     with respect to the SMART_ACTIONS constant. The additional index is used for
    #     the catergorical labeling of the actions as an "intermediate" solution for
    #     a restricted action space.
    #     """
    #     self.choice = self.epsilon_greedy()
    #     if self.choice == 'random' and agent_mode == 'learn':
    #         action_idx = np.random.randint(len(SMART_ACTIONS))  # not clean
    #         x_coord = np.random.randint(self.x_map_dim)
    #         y_coord = np.random.randint(self.y_map_dim)
    #         chosen_action = self.get_action(available_actions, SMART_ACTIONS[action_idx], x_coord, y_coord)
    #     else:
    #         with torch.no_grad():
    #             action_q_values, x_coord_q_values, y_coord_q_values = self.net(self.state_history_tensor)
    #         # action_idx = np.argmax(action_q_values)
    #         # print(x_coord_q_values.requires_grad)
    #         # print(x_coord_q_values.grad_fn)
    #         # print(x_coord_q_values)
    #         # print(y_coord_q_values)
    #         action_idx = 0
    #         best_action = SMART_ACTIONS[action_idx]
    #         # with torch.no_grad
    #         # x_numpy = x_coord_q_values
    #         # y_numpy = y_coord_q_values
    #         # with grad tracking
    #         x_numpy = x_coord_q_values.detach().numpy()
    #         y_numpy = y_coord_q_values.detach().numpy()
    #         # print(x_numpy)
    #         # print(y_numpy)
    #         x_coord = np.argmax(x_numpy)
    #         y_coord = np.argmax(y_numpy)
    #         # print("Chosen X {}".format(x_coord))
    #         # print("Chosen Y {}".format(y_coord))
    #         # print("Beacon center {}".format(self.beacon_center))
    #         chosen_action = self.get_action(available_actions, best_action, x_coord, y_coord)
    #     # square brackets arounc chosen_action needed for internal pysc2 state machine
    #     return [chosen_action], torch.tensor([action_idx], dtype=torch.long, device=self.device), torch.tensor([x_coord], dtype=torch.long, device=self.device), torch.tensor([y_coord], dtype=torch.long, device=self.device)

    # def save_data(self, obs):
    #     """
    #     Save the actual information in the history.
    #     As soon as there has been enough data, the experience is sampled from the replay buffer.
    #     """
    #     # collect transition data
    #     # print(self.beacon_center, self.player_center)
    #     # reward_shaped = 1
    #     # if obs.reward == 0:
    #     #     if math.isnan(self.beacon_center) or math.isnan(self.player_center):
    #     #         reward_x = 0
    #     #     else:
    #     #         reward_x = np.square(self.beacon_center[0] - self.player_center[0])
    #     #     if math.isnan(self.beacon_center) or math.isnan(self.player_center):
    #     #         reward_y = 0
    #     #     else:
    #     #         reward_y = np.square(self.beacon_center[1] - self.player_center[1])
    #     #     reward_shaped = - np.sqrt(reward_x + reward_y)/1000
    #     # # print("Reward X: {}, Reward Y: {}, Reward TOTAL: {}".format(reward_x, reward_y, reward_shaped))
    #     # reward = torch.tensor([reward_shaped], device=self.device, dtype=torch.float)
    #     # print(reward)
    #     reward = torch.tensor([obs.reward], device=self.device, dtype=torch.float)
    #     step_type = torch.tensor([obs.step_type], device=self.device, dtype=torch.int)
    #     next_state = torch.tensor([obs.observation.feature_screen.player_relative], dtype=torch.float, device=self.device, requires_grad = True)
    #
    #     # push next state on next state history stack
    #     # next_state_history_tensor = self.hist.stack(next_state)
    #     next_state_history_tensor = next_state
    #
    #     # save transition tuple to the memory buffer
    #
    #     # print(self.state_history_tensor)
    #     # print(self.action_idx, self.x_coord, self.y_coord)
    #     # print(next_state_history_tensor)
    #     # print(reward)
    #
    #     self.memory.push(self.state_history_tensor, self.action_idx, self.x_coord, self.y_coord, reward, next_state_history_tensor, step_type)
