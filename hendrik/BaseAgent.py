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
from AtariNet import DQN, SingleDQN

### torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# constants
SMART_ACTIONS = [
          actions.FUNCTIONS.select_army.id,
          actions.FUNCTIONS.Move_screen.id
        ]


class BaseAgent(base_agent.BaseAgent):
  def __init__(self, screen_dim, minimap_dim, batch_size, target_update_period,
                history_length):
    super(BaseAgent, self).__init__()
    self.history_length = history_length
    self.net, self.target_net, self.optimizer = self._build_model()
    self.map_dimensions = (84,64)
    self.screen_dim = screen_dim
    self.minimap_dim = minimap_dim
    self.epsilon = 1.0
    self.eps_start = 1.0
    self.eps_end = 0.1
    self.eps_decay = 3500
    self.steps_done = 0
    self.gamma = 0.9
    self.timesteps = 0
    self.update_cnt = 0
    self.target_update_period = target_update_period
    self.choice = None
    print("Network: \n{}".format(self.net))
    print("Optimizer: \n{}".format(self.optimizer))
    print("Target Network: \n{}".format(self.target_net))

  def _build_model(self):
    net = DQN(self.history_length)
    target_net = DQN(self.history_length)
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

  def can_do(self, available_actions, action):
    '''
    shortcut for checking if action is available at the moment

    '''
    return action in available_actions


  def get_action(self, available_actions, action, x, y):
    '''
    gets action from id list and passes valid args
    TODO: make this more intuitive and general
    '''
    if self.can_do(available_actions, action):
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


  def choose_action(self, available_actions, history_tensor):
    '''
    chooses an action according to the current policy
    returns the chosen action id with x,y coordinates and the index of the action
    with respect to the SMART_ACTIONS constant. The additional index is used for
    the catergorical labeling of the actions as an "intermediate" solution for
    a restricted action space.
    '''
    self.choice = self.epsilon_greedy()
    if self.choice=='random':
      action_idx = np.random.randint(len(SMART_ACTIONS))
      x_coord = np.random.randint(self.map_dimensions[0])
      y_coord = np.random.randint(self.map_dimensions[1])
      chosen_action = self.get_action(available_actions, SMART_ACTIONS[action_idx], x_coord , y_coord)
    else:
      with torch.no_grad():
        action_q_values, x_coord_q_values, y_coord_q_values = self.net(history_tensor)
      action_idx = np.argmax(action_q_values)
      best_action = SMART_ACTIONS[action_idx]
      x_coord = np.argmax(x_coord_q_values)
      y_coord = np.argmax(y_coord_q_values)
      chosen_action = self.get_action(available_actions, best_action, x_coord, y_coord)

    # square brackets arounc chosen_action needed for internal pysc2 state machine
    return [chosen_action], torch.tensor([action_idx],dtype=torch.long), \
        torch.tensor([x_coord],dtype=torch.long), \
        torch.tensor([y_coord],dtype=torch.long)

  def step(self, obs, env, history_tensor):
    self.timesteps += 1
    self.steps += 1

    # get dimensions from observation
    # self.x_map_dim, self.y_map_dim = obs.observation["feature_screen"]["player_relative"].shape


    action,  action_idx, x_coord,y_coord  =self.choose_action(obs.observation.available_actions, history_tensor)
    next_state = env.step(action)
    self.reward += obs.reward
    return next_state, action, action_idx, x_coord, y_coord


  def reset(self):
    super(BaseAgent, self).reset()
    self.timesteps = 0
    self.update_cnt = 0

  def update_target_net(self, src_net, target_net):
    '''
    updates weights of the target network, i.e. copies model weights to it
    '''
    target_net.load_state_dict(src_net.state_dict())
    self.update_cnt += 1
    self.update_status = "Weights updated!"




  def optimize(self, batch):
    '''
    optimizes the model. currently only trains the actions
    # TODO: extend Q-Update function to the x and y coordinates
     '''
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
    loss = F.mse_loss(q_values_cat, td_target_cat.unsqueeze(1))

    # optimize model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


    return loss.item()




if __name__ == '__main__':
  BaseAgent = BaseAgent()