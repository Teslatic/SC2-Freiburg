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


### custom imports
from AtariNet import DQN, ReplayBuffer


### torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# constants
SMART_ACTIONS = [   actions.FUNCTIONS.Move_screen.id,
                    actions.FUNCTIONS.select_army.id
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
        self.net, self.optimizer = self._build_model()

        self.state_q_value = torch.zeros(batch_size, device="cpu", requires_grad=True)
        self.td_target = torch.zeros(batch_size, device="cpu", requires_grad=True)

    def _build_model(self):
        net = DQN()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        return net, optimizer

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
        '''
        choice = self.epsilon_greedy()
        if choice=='random':
            chosen_action = self.get_action(obs, random.choice(SMART_ACTIONS), np.random.randint(0,83), np.random.randint(0,83))
        else:
            with torch.no_grad():
                action_q_values, x_coord_q_values, y_coord_q_values = \
                    self.net((obs.observation["feature_screen"]["player_relative"].reshape((-1,1,84,84))))
            best_action = SMART_ACTIONS[torch.argmax(action_q_values)]
            best_x = torch.argmax(x_coord_q_values).numpy()
            best_y = torch.argmax(y_coord_q_values).numpy()
            chosen_action = self.get_action(obs, best_action, best_x, best_y)

        return chosen_action


    def step(self, obs):
        '''
        takes one step in the internal state machine
        '''
        super(BaseAgent, self).step(obs)
        action = self.choose_action(obs)

        return action


    def train(self, batch):

        with torch.no_grad():
            for idx, (state, action, reward, next_state, step_type) in enumerate(batch):
                tmp_q , _, _, =  self.net(state.reshape((-1,1,84,84)))
                self.state_q_value[idx] = torch.tensor(torch.max(tmp_q),dtype=torch.float, requires_grad=True)
                # state_q_value[idx] =state_q_value)

                self.next_state_q_value, _, _ = self.net(next_state.reshape((-1,1,84,84)))
                if step_type==StepType.LAST:
                    # dtype casting necessary since network output is float32 and reward is int64
                    self.td_target[idx] = torch.tensor(reward, dtype=torch.float, requires_grad=True)
                else:
                    # dtype casting necessary since network output is float32 and reward is int64
                    self.td_target[idx] = torch.tensor((reward + self.gamma * torch.max(self.next_state_q_value)),dtype=torch.float, requires_grad=True)
                    print(self.td_target[idx])

            loss = F.mse_loss(self.state_q_value, self.td_target)
        self.optimizer.zero_grad()
        loss.backward()




if __name__ == '__main__':
    BaseAgent = BaseAgent()
