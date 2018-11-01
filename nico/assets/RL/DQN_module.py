import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple

from AtariNet import DQN
from assets.memory.ReplayBuffer import ReplayBuffer
from assets.helperFunctions.timestamps import print_timestamp as print_ts

class DQN_module():
    """
    A wrapper class that augments the AtariNet.DQN class by optmizing methods.
    """
    def __init__(self, batch_size, gamma, history_length, size_replaybuffer, optim_learning_rate):
        """
        update_cnt: Target network update counter.
        net:        The q-value network.
        target_net: The target network.
        optimizer:  The optimizer used to update the q-value network.
        memory:     The memory class for the replay buffer
        """
        self.device = self._setup_torch()
        self.batch_size = batch_size
        self.gamma = gamma
        self.history_length = history_length
        self.size_replaybuffer = size_replaybuffer
        self.optim_learning_rate = optim_learning_rate
        self.loss = 0
        self.update_cnt = 0  # Target network update_counter

        self.net, self.target_net, self.optimizer = self._build_model()
        self.print_architecture()
        self.memory = ReplayBuffer(self.size_replaybuffer)

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

    def _setup_torch(self):
        """
        Setting GPU if available. Else, use the CPU.
        """
        # Initalizing
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_printoptions(linewidth=750, profile="full")
        print_ts("Performing calculations on {}".format(device))
        return device

    def predict_q_values(self, state):
        state_tensor = torch.tensor([state], device=self.device, dtype=torch.float, requires_grad=False)
        return self.net(state_tensor)

    # ##########################################################################
    # Optimizing the network
    # ##########################################################################

    def optimize(self):
        """
        optimizes the model. currently only trains the actions
        # TODO: extend Q-Update function to the x and y coordinates
        """

        # Sample and unzip batch from experience replay buffer
        batch = self.sample_batch()
        self.unzip_batch(batch)

        # Calculate Q-values
        self.calculate_q_values()

        # calculate td targets of the actions, x&y coordinates
        self.td_target = self.calculate_td_target(self.next_state_q_values_max)

        # Compute the loss
        self.compute_loss()

        # optimize model
        self.optimize_model()

        return self.loss

    def sample_batch(self):
        """
        Sample from batch.
        """
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        transitions = self.memory.sample(self.batch_size)
        return Transition(*zip(*transitions))

    def unzip_batch(self, batch):
        """
        Get the batches from the transition tuple
        np.concatenate concatenate all the batch data into a single ndarray
        """
        self.state_batch =  torch.as_tensor(np.concatenate([batch.state]), device=self.device, dtype=torch.float)
        # print(self.state_batch)
        self.action_batch = torch.as_tensor(batch.action, device=self.device)
        # print(self.action_batch)
        self.reward_batch = torch.as_tensor(batch.reward, device=self.device, dtype=torch.float)
        # print(self.reward_batch)
        self.next_state_batch = torch.as_tensor(np.concatenate([batch.next_state]), device=self.device, dtype=torch.float)
        # print(self.next_state_batch)

    def calculate_q_values(self):
        """
        """
        # forward pass
        self.state_q_values_full = self.net(self.state_batch)

        # gather action values with respect to the chosen action
        self.state_q_values = self.state_q_values_full.gather(1, self.action_batch)

        # compute action values of the next state over all actions and take the max
        self.next_state_q_values = self.target_net(self.next_state_batch)
        self.next_state_q_values_max = self.next_state_q_values.max(1)[0].detach()

    def calculate_td_target(self, q_values_best_next):
        """
        Calculating the TD-target for given q-values.
        """
        return (q_values_best_next * self.gamma) + self.reward_batch

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
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        """
        updates weights of the target network, i.e. copies model weights to it
        """
        self.transfer_weights(self.net, self.target_net)
        self.update_cnt += 1

    # ##########################################################################
    # Weight methods
    # ##########################################################################

    def transfer_weights(self, src_net, target_net):
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
