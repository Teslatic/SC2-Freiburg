import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from pysc2.agents import base_agent
from pysc2.lib import actions, features
from collections import namedtuple

from AtariNet import DQN
from assets.smart_actions import SMART_ACTIONS_MOVE2BEACON as SMART_ACTIONS
from assets.memory.ReplayBuffer import ReplayBuffer
from assets.memory.History import History
np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)  # only for terminal output visualization


class BaseAgent(base_agent.BaseAgent):
    """
    explanation missing
    """
    def __init__(self, agent_file):
        super(BaseAgent, self).__init__()
        self.unzip_hyperparameter_file(agent_file)

        self.memory = ReplayBuffer(self.size_replaybuffer)
        self.hist = History(self.history_length)
        self.net, self.target_net, self.optimizer = self._build_model()
        self.print_architecture()

    def unzip_hyperparameter_file(self, agent_file):
        self.screen_dim = agent_file['SCREEN_DIM']
        self.minimap_dim = agent_file['MINIMAP_DIM']
        self.x_map_dim = 84  # magic number
        self.y_map_dim = 64  # magic number
        self.map_dimensions = (self.x_map_dim , self.y_map_dim)
        self.gamma = agent_file['GAMMA']
        self.optim_learning_rate = agent_file['OPTIM_LR']
        self.batch_size = agent_file['BATCH_SIZE']
        self.target_update_period = agent_file['TARGET_UPDATE_PERIOD']
        self.history_length = agent_file['HIST_LENGTH']
        self.size_replaybuffer = agent_file['REPLAY_SIZE']
        self.device = agent_file['DEVICE']

        self.epsilon = 1.0
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay = 20000

        self.steps_done = 0
        self.timesteps = 0
        self.update_cnt = 0
        self.choice = None
        self.update_status = '-'
        self.loss = 0

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
                            use_feature_units = True)
        return self.agent_interface

    def _build_model(self):
        """
        Initializing 2 networks and an Adam optimizer.
        """
        net = DQN(self.history_length)
        target_net = DQN(self.history_length)
        optimizer = optim.Adam(net.parameters(), lr=self.optim_learning_rate)
        return net, target_net, optimizer

    def print_architecture(self):
        print("Network: \n{}".format(self.net))
        print("Optimizer: \n{}".format(self.optimizer))
        print("Target Network: \n{}".format(self.target_net))

    def print_status(self):
        """
        function helper for status printing
        """
        print("Epsilon: {:.2f}\t| choice: {}".format(self.epsilon, self.choice))
        print("Episode {}\t| Step {}\t| Total Steps: {}".format(self.episodes, self.timesteps, self.steps))
        print("Chosen action: {}".format(self.action))
        print("chosen coordinates [x,y]: {}".format((self.x_coord.item(), self.y_coord.item())))
        print("Beacon center location [x,y]: {}".format(self.beacon_center))
        print("Current Episode Score: {}\t| Total Score: {}".format(self.last_score, self.reward))
        print("Action Loss: {:.5f}".format(self.loss))
        print("{}".format(self.update_status))
        print("----------------------------------------------------------------")

    def unit_type_is_selected(self, obs, unit_type):
        """
        Check if certain unit type is selected at moment of call, supports single
        select and multi_select
        """
        if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False

    def get_unit_by_type(self, obs, unit_type):
        """
        select units by type, i.e. CTRL + LEFT_CLICK (on screen)
        """
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def can_do(self, available_actions, action):
        """
        shortcut for checking if action is available at the moment
        """
        return action in available_actions

    def get_action(self, available_actions, action, x, y):
        """
        gets action from id list and passes valid args
        TODO: make this more intuitive and general
        """
        if self.can_do(available_actions, action):
            if action == actions.FUNCTIONS.Move_screen.id:
                # print("Move_screen action was selected")
                return actions.FUNCTIONS.Move_screen("now", (x,y))
            if action == actions.FUNCTIONS.select_army.id:
                # print("Select_army action was selected")
                return actions.FUNCTIONS.select_army("select")
            if action == actions.FUNCTIONS.no_op.id:
                # print("No_op action was selected")
                return actions.FUNCTIONS.no_op()
        else:
            # print("No_op action was selected because selected agent was not available")
            return actions.FUNCTIONS.no_op()

    def epsilon_greedy(self):
        """
        returns a string in order to determine if the next action choice is
        going to be random or according to an decaying epsilon greeedy policy
        """
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        choice = np.random.choice(['random','greedy'], p = [self.epsilon,1-self.epsilon])
        return choice

    def choose_action(self, available_actions):
        """
        chooses an action according to the current policy
        returns the chosen action id with x,y coordinates and the index of the action
        with respect to the SMART_ACTIONS constant. The additional index is used for
        the catergorical labeling of the actions as an "intermediate" solution for
        a restricted action space.
        """
        self.choice = self.epsilon_greedy()
        if self.choice == 'random':
            action_idx = np.random.randint(len(SMART_ACTIONS))  # not clean
            x_coord = np.random.randint(self.x_map_dim)
            y_coord = np.random.randint(self.y_map_dim)
            chosen_action = self.get_action(available_actions, SMART_ACTIONS[action_idx], x_coord , y_coord)
        else:
            with torch.no_grad():
                action_q_values, x_coord_q_values, y_coord_q_values = self.net(self.history_tensor)
            action_idx = np.argmax(action_q_values)
            best_action = SMART_ACTIONS[action_idx]
            x_coord = np.argmax(x_coord_q_values)
            y_coord = np.argmax(y_coord_q_values)
            chosen_action = self.get_action(available_actions, best_action, x_coord, y_coord)
        # square brackets arounc chosen_action needed for internal pysc2 state machine
        return [chosen_action], torch.tensor([action_idx], dtype=torch.long), torch.tensor([x_coord], dtype=torch.long), torch.tensor([y_coord],dtype=torch.long)

    def initializing_timestep(self, beacon_center, last_score):
        """
        Initializing variables and flags
        """
        self.beacon_center = beacon_center
        self.timesteps += 1
        self.steps += 1
        # self.actual_reward = obs.reward
        self.last_score = last_score

    def step(self, obs, beacon_center, last_score):
        """
        Choosing an action
        """
        self.reward += obs.reward
        self.step_type = obs.step_type
        self.available_actions = obs.observation.available_actions
        self.initializing_timestep(beacon_center, last_score)
        self.state = torch.tensor([obs.observation.feature_screen.player_relative], dtype=torch.float)
        self.history_tensor = self.hist.stack(self.state)  # stack state on history
        self.state_history_tensor = self.state.unsqueeze(1)
        self.action, self.action_idx, self.x_coord, self.y_coord = self.choose_action(self.available_actions)
        self.print_status()
        print("Saving episode data")
        self.save_data(obs)

        if len(self.memory) >= self.batch_size:
            print("Optimize on sampled batch. Calculating loss")
            self.loss = self.optimize(self.sample_batch())

        # check if done, i.e. step_type==2
        if self.step_type == 2:
            # update target nets
            print("About to update")
            if self.episodes % self.target_update_period == 0 and self.episodes != 0:
                self.update_target_net(self.net, self.target_net)
            self.reset()
        # print(self.action)
        return self.action

    def sample_batch(self):
        """
        Sample from batch.
        """
        Transition = namedtuple('Transition', ('state', 'action', 'x_coord', 'y_coord', 'reward', 'next_state','step_type'))
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
        self.update_status = "Weights updated!"

    def save_data(self, obs):
        """
        Save the actual information in the history.
        As soon as there has been enough data, the experience is sampled from the replay buffer.
        """
        # collect transition data
        reward = torch.tensor([obs.reward], device=self.device, dtype=torch.float)
        step_type = torch.tensor([obs.step_type], device=self.device, dtype=torch.int)
        next_state = torch.tensor([obs.observation.feature_screen.player_relative], dtype=torch.float)

        # push next state on next state history stack
        next_state_history_tensor = self.hist.stack(next_state)

        # save transition tuple to the memory buffer
        self.memory.push(self.state_history_tensor, self.action_idx, self.x_coord, self.y_coord, reward, next_state_history_tensor, step_type)

    def optimize(self, batch):
        """
        optimizes the model. currently only trains the actions
        # TODO: extend Q-Update function to the x and y coordinates
        """
        # get the batches from the transition tuple
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        x_coord_batch = torch.cat(batch.x_coord).unsqueeze(1)
        y_coord_batch = torch.cat(batch.y_coord).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)
        step_type_batch = torch.cat(batch.step_type)
        next_state_batch = torch.cat(batch.next_state)

        # forward pass
        state_action_values, x_q_values, y_q_values = self.net(state_batch)

        # gather action values with respect to the chosen action
        state_action_values = state_action_values.gather(1,action_batch)
        x_q_values = x_q_values.gather(1,x_coord_batch)
        y_q_values = y_q_values.gather(1,y_coord_batch)
        # compute action values of the next state over all actions and take the max
        next_state_values, next_x_q_values, next_y_q_values = self.target_net(next_state_batch)

        next_state_values = next_state_values.max(1)[0].detach()
        next_x_q_values = next_x_q_values.max(1)[0].detach()
        next_y_q_values = next_y_q_values.max(1)[0].detach()

        # calculate td targets of the actions
        td_target_actions = torch.tensor((next_state_values * self.gamma) + reward_batch , dtype=torch.float)
        td_target_actions[np.where(step_type_batch==2)] = reward_batch[np.where(step_type_batch==2)]

        # calculate td targets of the x coord
        td_target_x_coord = torch.tensor((next_x_q_values * self.gamma) + reward_batch , dtype=torch.float)
        td_target_x_coord[np.where(step_type_batch==2)] = reward_batch[np.where(step_type_batch==2)]

        # calculate td targets of the y coord
        td_target_y_coord = torch.tensor((next_y_q_values * self.gamma) + reward_batch , dtype=torch.float)
        td_target_y_coord[np.where(step_type_batch==2)] = reward_batch[np.where(step_type_batch==2)]

        q_values_cat = torch.cat((state_action_values, x_q_values, y_q_values))
        td_target_cat = torch.cat((td_target_actions, td_target_x_coord, td_target_y_coord))

        # compute MSE loss
        self.loss = F.mse_loss(q_values_cat, td_target_cat.unsqueeze(1))

        # optimize model
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

if __name__ == '__main__':
  BaseAgent = BaseAgent()
