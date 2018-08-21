### pysc imports
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.env.environment import StepType

### normal python modules
import random
import numpy as np
np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan) # only for terminal ouput visualization
from absl import app
from sklearn import preprocessing

### custom imports
from AtariNet import DQN


### torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# constants
SMART_ACTIONS = [   actions.FUNCTIONS.no_op.id,
                    actions.FUNCTIONS.select_army.id,
                    actions.FUNCTIONS.Move_screen.id
                ]


class BaseAgent(base_agent.BaseAgent):
    def __init__(self, screen_dim, minimap_dim, batch_size):
        super(BaseAgent, self).__init__()
        self.screen_dim = screen_dim - 1        # -1 for array indexing
        self.minimap_dim = minimap_dim - 1      # -1 for array indexing
        self.epsilon = 1.0
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 500
        self.steps_done = 0
        self.gamma = 0.9

        self.net, self.target_net, self.optimizer = self._build_model()
        print("Network: \n{}".format(self.net))
        print("Optimizer: \n{}".format(self.optimizer))
        print("Target Network: \n{}".format(self.target_net))
        self.state_q_value = torch.zeros(batch_size, device="cpu", requires_grad=True)
        self.td_target = torch.zeros(batch_size, device="cpu", requires_grad=True)

    def _build_model(self):
        net = DQN()
        target_net = DQN()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        return net, target_net, optimizer

    def unit_type_is_selected(self, obs, unit_type):
        '''
        check if certain unit type is selected at moment of call, supports single
        select and multi_select
        '''
        if (len(obs.observation.single_select) > 0 and
        obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
        obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False


    def get_unit_by_type(self, obs, unit_type):
        '''
        select units by type, i.e. CTRL + LEFT_CLICK (on screen)
        '''
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        '''
        shortcut for checking if action is available at the moment
        '''
        return action in obs.observation.available_actions

    def get_action(self, obs, action, x, y):
        '''
        gets action from id list and passes valid args
        TODO: make this more intuitive and general
        '''
        if self.can_do(obs, action):
            if action == actions.FUNCTIONS.Move_screen.id:
                return actions.FUNCTIONS.Move_screen("now", (x,y))
            if action == actions.FUNCTIONS.select_army.id:
                return actions.FUNCTIONS.select_army("select")
            if action == actions.FUNCTIONS.no_op.id:
                return actions.FUNCTIONS.no_op()
        else:
            return actions.FUNCTIONS.no_op()

    def epsilon_greedy(self):
        '''
        returns a string in order to determine if the next action choice is
        going to be random or according to an decaying epsilon greeedy policy
        '''
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) \
                        * np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        choice = np.random.choice(['random','greedy'], p = [self.epsilon,1-self.epsilon])
        return choice


    def choose_action(self, obs):
        '''
        chooses an action according to the current policy
        returns the chosen action id with x,y coordinates and the index of the action
        with respect to the SMART_ACTIONS constant. The additional index is used for
        the catergorical labeling of the actions as an "intermediate" solution for
        a restricted action space.
        '''
        choice = self.epsilon_greedy()
        if choice=='random':
            action_idx = np.random.randint(len(SMART_ACTIONS))
            chosen_action = self.get_action(obs, SMART_ACTIONS[action_idx], torch.randint(83,(1,), dtype=torch.long), torch.randint(83,(1,), dtype=torch.long))
        else:
            with torch.no_grad():
                net_input = torch.tensor((
                    obs.observation["feature_screen"]["player_relative"].reshape((-1,1,84,84)))
                    ,dtype=torch.float)
                action_q_values, x_coord_q_values, y_coord_q_values = \
                    self.net(net_input)
            action_idx = torch.argmax(action_q_values)
            best_action = SMART_ACTIONS[action_idx]
            best_x = torch.argmax(x_coord_q_values).numpy()
            best_y = torch.argmax(y_coord_q_values).numpy()
            chosen_action = self.get_action(obs, best_action, best_x, best_y)

        # square brackets needed for internal pysc2 state machine
        return [chosen_action], torch.tensor([action_idx],dtype=torch.long)

    def step(self, obs):
        '''
        takes one step in the internal state machine
        '''
        super(BaseAgent, self).step(obs)
        action, action_idx = self.choose_action(obs)

        return action, action_idx


    def optimize(self, batch):
        '''
        optimizes the model. currently only trains the actions
        # TODO: extend Q-Update function to the x and y coordinates
       '''

        # get the batches from the transition tuple
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)
        step_type_batch = torch.cat(batch.step_type)
        next_state_batch = torch.cat(batch.next_state)

        # gather action values with respect to the chosen action
        state_action_values = self.net(state_batch.reshape((-1,1,84,84)))[0].gather(1,action_batch)

        # compute action values of the next state over all actions and take the max
        next_state_values, _, _ = self.target_net(next_state_batch.reshape((-1,1,84,84)))
        next_state_values = next_state_values.max(1)[0].detach()

        # calculate td targets
        td_target = torch.tensor((next_state_values * self.gamma) + reward_batch , dtype=torch.float)
        td_target[np.where(step_type_batch==2)] = reward_batch[np.where(step_type_batch==2)]

        # compute MSE loss
        loss = F.mse_loss(state_action_values, td_target.unsqueeze(1))

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target weights
        self.target_net.load_state_dict(self.net.state_dict())

        return loss.item()


if __name__ == '__main__':
    BaseAgent = BaseAgent()
