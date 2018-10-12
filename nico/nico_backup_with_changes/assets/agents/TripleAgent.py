# pysc imports
from pysc2.agents import base_agent
# from pysc2.env import sc2_env
from pysc2.lib import actions, features

# normal python modules
import numpy as np

# custom imports
from AtariNet import SingleDQN

# torch imports
import torch
import torch.optim as optim
import torch.nn.functional as F

from assets.memory.ReplayBuffer import ReplayBuffer

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

# constants
SMART_ACTIONS = [
          actions.FUNCTIONS.select_army.id,
          actions.FUNCTIONS.Move_screen.id
        ]


class TripleAgent(base_agent.BaseAgent):
    """
    Alternative approach: use 3 networks, one for actions,x and y
    """
    def __init__(self, agent_file):
        super(TripleAgent, self).__init__()
        self.unzip_hyperparameter_file(agent_file)
        self.memory = ReplayBuffer(1000000)  # Put this into unzip
        self.action_net, self.action_target_net, self.action_optimizer = self._build_model(len(SMART_ACTIONS))  # NOT
        self.x_coord_net, self.x_coord_target_net, self.x_coord_optimizer = self._build_model(84)
        self.y_coord_net, self.y_coord_target_net, self.y_coord_optimizer = self._build_model(64)

    def unzip_hyperparameter_file(self, agent_file):
        self.screen_dim = agent_file['SCREEN_DIM']
        self.minimap_dim = agent_file['MINIMAP_DIM']
        self.x_map_dim = 84  # magic number
        self.y_map_dim = 64  # magic number
        self.map_dimensions = (self.x_map_dim , self.y_map_dim)
        self.target_update_period = agent_file['TARGET_UPDATE_PERIOD']
        self.history_length = agent_file['HIST_LENGTH']

        self.epsilon = 1.0
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay = 2800
        self.steps_done = 0
        self.gamma = 0.99999999
        self.timesteps = 0
        self.update_cnt = 0
        self.choice = None

    def setup_interface(self):
        self.agent_interface = features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=self.screen_dim,
                            minimap=self.minimap_dim),
                            # rgb_dimensions = features.Dimensions(screen=SCREEN_DIM,
                            #                     minimap=MINIMAP_DIM),
                            # action_space = actions.ActionSpace.RGB,
                            use_feature_units = True)
        return self.agent_interface

    def _build_model(self, num_outputs):
        """
        Initializing 2 networks and an Adam optimizer.
        """
        net = SingleDQN(self.history_length, num_outputs)
        target_net = SingleDQN(self.history_length, num_outputs)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        return net, target_net, optimizer

    def print_status(self, env, action, x_coord, y_coord, beacon_center, action_loss, x_coord_loss, y_coord_loss):
        '''
        function helper for status printing
        '''
        print("Epsilon: {:.2f}\t| choice: {}".format(self.epsilon, self.choice))
        print("Episode {}\t| Step {}\t| Total Steps: {}".format(self.episodes, self.timesteps, self.steps))
        print("Chosen action: {}".format(action))
        print("chosen coordinates [x,y]: {}".format((x_coord, y_coord)))
        print("Beacon center location [x,y]: {}".format(beacon_center))
        print("Current Episode Score: {}\t| Total Score: {}".format(env._last_score[0],self.reward))
        print("Action Loss: {:.5f}\t| X coord loss: {:.5f}\t| Y coord loss: {:.5f}".format(action_loss, x_coord_loss, y_coord_loss))
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
                action_q_values = self.action_net(history_tensor)
                x_coord_q_values = self.x_coord_net(history_tensor)
                y_coord_q_values = self.y_coord_net(history_tensor)
            action_idx = np.argmax(action_q_values)
            best_action = SMART_ACTIONS[action_idx]
            x_coord = np.argmax(x_coord_q_values)
            y_coord = np.argmax(y_coord_q_values)
            chosen_action = self.get_action(available_actions, best_action, x_coord, y_coord)

        # square brackets arounc chosen_action needed for internal pysc2 state machine
        return [chosen_action], torch.tensor([action_idx],dtype=torch.long), \
            torch.tensor([x_coord],dtype=torch.long), \
            torch.tensor([y_coord],dtype=torch.long)

    def step(self, obs, env):
        self.timesteps += 1
        self.steps += 1
        self.reward += obs.reward

        # get dimensions from observation
        # self.x_map_dim, self.y_map_dim = obs.observation["feature_screen"]["player_relative"].shape
        self.action, self.action_idx, self.x_coord, self.y_coord = self.choose_action(obs.observation.available_actions, self.history_tensor)
        self.next_state = env.step(action)



        # collect transition data
        reward = torch.tensor([actual_obs.reward], device=device,dtype=torch.float)
        step_type = torch.tensor([actual_obs.step_type], device=device,dtype=torch.int)
        next_state = torch.tensor([next_obs.observation.feature_screen.player_relative],dtype=torch.float)

        # push next state on next state history stack
        next_state_history_tensor = hist.stack(next_state)

        # save transition tuple to the memory buffer
        memory.push(state_history_tensor, action_idx, x_coord, y_coord, reward, next_state_history_tensor, step_type)

        ########
        # OPTIMIZE ON BATCH
        ########
        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)

            # zipping necessary for reasons i dont really understand
            batch = Transition(*zip(*transitions))

            # loss is not very expressive, but included just for sake of completeness
            action_loss, x_coord_loss, y_coord_loss = agent.optimize(batch)

          ########
          # UPDATE TARGET NETWORKS --> Put this into agent
          ########

                # update target nets
            if self.episodes % TARGET_UPDATE_PERIOD == 0:
              self.update_target_net(agent.action_net, agent.action_target_net)
              self.update_target_net(agent.x_coord_net, agent.x_coord_target_net)
              self.update_target_net(agent.y_coord_net, agent.y_coord_target_net)
            else:
              self.update_status = " - "

        return self.next_state, action, action_idx, x_coord, y_coord


    def reset(self):
        super(TripleAgent, self).reset()
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

        # forward pass and gather action values with respect to the chosen action
        state_action_values = self.action_net(state_batch).gather(1,action_batch)
        x_q_values = self.x_coord_net(state_batch).gather(1,x_coord_batch)
        y_q_values = self.y_coord_net(state_batch).gather(1,y_coord_batch)


        # compute action values of the next state over all actions and take the max
        next_state_action_values = self.action_target_net(state_batch).max(1)[0].detach()
        next_x_q_values = self.x_coord_target_net(state_batch).max(1)[0].detach()
        next_y_q_values = self.y_coord_target_net(state_batch).max(1)[0].detach()


        # calculate td targets of the actions
        td_target_actions = torch.tensor((next_state_action_values * self.gamma) + reward_batch , dtype=torch.float)
        td_target_actions[np.where(step_type_batch==2)] = reward_batch[np.where(step_type_batch==2)]

        # calculate td targets of the x coord
        td_target_x_coord = torch.tensor((next_x_q_values * self.gamma) + reward_batch , dtype=torch.float)
        td_target_x_coord[np.where(step_type_batch==2)] = reward_batch[np.where(step_type_batch==2)]

        # calculate td targets of the y coord
        td_target_y_coord = torch.tensor((next_y_q_values * self.gamma) + reward_batch , dtype=torch.float)
        td_target_y_coord[np.where(step_type_batch==2)] = reward_batch[np.where(step_type_batch==2)]

        q_values_cat = torch.cat((state_action_values, x_q_values, y_q_values))
        td_target_cat = torch.cat((td_target_actions, td_target_x_coord, td_target_y_coord))

        # compute MSE losses
        action_loss = F.mse_loss(state_action_values, td_target_actions.unsqueeze(1))
        x_coord_loss = F.mse_loss(x_q_values, td_target_x_coord.unsqueeze(1))
        y_coord_loss = F.mse_loss(y_q_values, td_target_y_coord.unsqueeze(1))

        # optimize models
        self.action_optimizer.zero_grad()
        action_loss.backward()
        self.action_optimizer.step()

        self.x_coord_optimizer.zero_grad()
        x_coord_loss.backward()
        self.x_coord_optimizer.step()

        self.x_coord_optimizer.zero_grad()
        y_coord_loss.backward()
        self.y_coord_optimizer.step()


        return action_loss.item(), x_coord_loss.item(), y_coord_loss.item()
