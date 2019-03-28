import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from assets.hdrln.Controller import Controller
from assets.hdrln.DeepSkillNetwork import DeepSkillNetwork
from assets.hdrln.SkillExperienceReplayBuffer import SERB
from assets.hdrln.StudentNetwork import StudentNetwork

from collections import namedtuple

from assets.helperFunctions.timestamps import print_timestamp as print_ts


class HDRLN_module():
    """
    A wrapper class that contains all the HDRLN functionality.
    """
    ###########################################################################
    # Initializing
    ###########################################################################
    def __init__(self, module_specs):
        """
        Setting up the needed parameters for the HDRLN module.
        Setting up:
        - DeepSkillNetwork
        - Controller
        - Skill Experience Replay Buffer
        - StudentNetwork (distilled Neural Network)
        """
        # TODO (NOT): ACTION SPACE DEFINED A PRIORI--> Primitive action skill module???
        self.DSN_mode = 'array'
        self.device = self._setup_torch('cuda:0')

        self.unzip_module_specs(module_specs)
        self.DSN = DeepSkillNetwork()
        self.Controller = Controller()
        self.memory = SERB(1000)
        self.Student = StudentNetwork()

    def unzip_module_specs(self, module_specs):
        """
        """
        self.batch_size = module_specs["BATCH_SIZE"]
        self.gamma = module_specs["GAMMA"]
        self.history_length = module_specs["HISTORY_LENGTH"]
        self.dim_actions = module_specs["DIM_ACTIONS"]
        self.size_replaybuffer = module_specs["SIZE_REPLAYBUFFER"]
        self.optim_learning_rate = module_specs["OPTIM_LR"]
        self.loss = 0
        self.update_cnt = 0  # Target network update_counter

    def _setup_torch(self, GPU):
        """
        Setting GPU if available. Else, use the CPU.
        """
        # Initalizing
        device = torch.device(GPU if torch.cuda.is_available() else "cpu")
        torch.set_printoptions(linewidth=750, profile="full")
        print_ts("Performing calculations on {}".format(device))
        return device

    def add(self, skill_dir, skill_name_list, agent_list):
        """
        Wrapper for the DSN object of the HDRLN agent.
        Returns the number of registered skills of the agent.
        """
        self.N_skills = self.DSN.add(skill_dir, skill_name_list, agent_list)

    def seal_skill_policy(self):
        """
        Seals the number of skills for the skill policy and creates the skill
        policy neural network.
        """
        self.seal_skill_space()
        self.Controller.create_skill_policy()

    def seal_skill_space(self):
        self.N_skills = self.DSN.seal()
        self.Controller.seal_skills(self.N_skills)

    ###########################################################################
    # Policy
    ###########################################################################

    def policy(self, state, available_actions, mode):
        """
        The HDRLN policy consists of two stages:
        1. Skill policy: Selects the skill or primitive DQN module
        2. Action policy: Selects the Q-value of the selected skill
        """
        self.mode = mode
        self.Controller.set_available_actions(available_actions)

        # Stage 1: Choose a skill
        self.skill_idx = self.skill_policy(state)
        print("skill index:" + str(self.skill_idx))
        # Stage 2: Choose an action
        return self.action_policy(state, self.skill_idx)

    def skill_policy(self, state):
        """
        """
        if self.mode is 'distilling':  # Controller decides if skill or primitive action is used
        	return self.Controller.distill(state)
        elif self.mode is 'learning':
        	return self.Controller.epsilon_skill_policy(state)
        elif self.mode is 'testing':
        	return self.Controller.pick_skill_policy(state)
        else:
            print("Agent mode not known. No action selected")
            return 'no skill', 'no skill_idx'

    def action_policy(self, state, skill_idx):
        """
        """
        if self.DSN_mode is 'array':
            self.action_idx = self.DSN.pick(state, skill_idx)
        elif self.DSN_mode is 'distilled':
            self.action_idx = self.Student.pick(state)
        return self.action_idx

    def optimize(self):
        """
        Optimizes the HDRLN module on a minibatch
        """
        # Sample and unzip batch from experience replay buffer
        batch = self.sample_batch()
        self.unzip_batch(batch)

        # Calculate Q-values
        self.calculate_q_values()

        # calculate td targets of the actions, x&y coordinates
        self.td_target = self.calculate_td_target(self.next_state_q_max)

        # Compute the loss
        self.compute_loss()

        # optimize model
        self.optimize_model()

        return self.loss

    def sample_batch(self):
        """
        Sample from batch.
        """
        Transition = namedtuple('Transition',
                                ('state', 'action', 'reward', 'next_state'))
        transitions = self.memory.sample(self.batch_size)
        return Transition(*zip(*transitions))

    def unzip_batch(self, batch):
        """
        Get the batches from the transition tuple
        np.concatenate concatenate all the batch data into a single ndarray
        """
        self.state_batch = torch.as_tensor(np.concatenate([batch.state]),
                                           device=self.device,
                                           dtype=torch.float)
        self.action_batch = torch.tensor(batch.action,
                                         device=self.device,
                                         dtype=torch.long)
        self.reward_batch = torch.as_tensor(batch.reward,
                                            device=self.device,
                                            dtype=torch.float)
        self.next_state_batch = torch.as_tensor(
                                            np.concatenate([batch.next_state]),
                                            device=self.device,
                                            dtype=torch.float)


        self.state_batch = self.state_batch.squeeze(1)
        self.next_state_batch = self.next_state_batch.squeeze(1)

    def calculate_q_values(self):
        """
        """
        # forward pass
        self.state_q_values = self.net(self.state_batch)

        # gather action values with respect to the chosen action
        self.state_q_values = self.state_q_values.gather(1, self.action_batch)

        # compute action values of the next state over all actions and take max
        self.next_state_q_values = self.target_net(self.next_state_batch)
        self.next_state_q_max = self.next_state_q_values.max(1)[0].detach()

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
        self.loss = F.mse_loss(self.state_q_values,
                               self.td_target.unsqueeze(1))

    # ##########################################################################
    # Evaluate a timestep
    # ##########################################################################

    def optimize_model(self):
        """
        This step is used to update the parameter weights:
        loss.backwards propagates the backward gradients for all net weights.
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

    def update_student_network(self):
        """
        """
        self.Student.update()
