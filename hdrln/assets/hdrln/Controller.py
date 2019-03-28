import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple

from assets.RL.DQN_module import DQN_module

from assets.helperFunctions.timestamps import print_timestamp as print_ts

class Controller():
    """
    The Controller class is used to learn, when to use which skill.
    It is the heart of the skill policy.
    A controller has to be learned for every environment.
    """
    ###########################################################################
    # Initializing
    ###########################################################################

    def __init__(self):
        self.lock = False
        self.locked_skill_idx = -1
        self.distill_counter = -1
        self.N_skills = 0

        self.eps_start = 1
        self.eps_decay = 8000
        self.eps_end = 0.01
        self.epsilon = 0
        self.steps_done = 0  # total_timesteps
        self.timesteps = 0  # timesteps in the current episode

    def seal_skills(self, N_skills):
        """
        Used to seal the number of available skills. This is used for round
        robin counting.
        """
        print_ts("Controller skills are sealed!")
        self.N_skills = N_skills

    def create_skill_policy(self, module_specs):
        """
        Creates the DQN module responsible for the skill policy.
        """
        self.net = DQN_module(module_specs)

    def set_available_actions(self, available_actions):
        """
        Setting the available actions for the skill policy
        """
        self.available_actions = available_actions


    def lock_skill(self, idx):
        self.lock = True
        self.locked_skill_idx = idx

    ###########################################################################
    # Policy
    ###########################################################################

    def distill_skill_policy(self, state):
        """
        1. Predict state with teacher and student network
        """
        self.distill_counter += 1
        if self.distill_counter == 50:
            self.distill_counter = 0
            self.skill_idx += 1
            if self.skill_idx > self.N_skills:
                self.skill_idx = 0
        return self.skill_idx

    def epsilon_skill_policy(self, state):
        """
        1. Epsilon greedy choice (random or pick)
        """
        # TODO (NOT): clean wrapup of algorithm
        self.choice = self.epsilon_greedy()
        print(self.epsilon, self.choice)
        if self.choice == 'random':
            skill_idx = np.random.randint(self.N_skills)
        else:
            skill_idx = self.pick_skill_policy()

        return skill_idx

        # if self.lock:
        #     self.skill_idx = self.locked_skill_idx
        #     return self.skill, self.skill_idx
        # return self.skill, self.skill_idx

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

    def pick_skill_policy(self):
        """
        1. Predict all Q
        2. Mask with available action
        3. Find the skill with the maximum Q-value
        4. Lock the skill
        """
        action_q_values = self.DQN.predict_q_values(self.state)
        # print(max(action_q_values[0]), 1/(84*64))        # Beste Action bestimmen
        best_action_numpy = action_q_values.detach().cpu().numpy()
        action_idx = np.argmax(best_action_numpy)
        best_action = self.action_space[action_idx]
        chosen_action = best_action
        return chosen_action, action_idx
