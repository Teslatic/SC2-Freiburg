from assets.hdrln.Controller import Controller
from assets.hdrln.DeepSkillNetwork import DeepSkillNetwork
from assets.hdrln.SkillExperienceReplayBuffer import SERB
from assets.hdrln.StudentNetwork	import StudentNetwork

from assets.helperFunctions.timestamps import print_timestamp as print_ts


class HDRLN_module():
    """
    A wrapper class that contains all the HDRLN functionality.
    """
    ###########################################################################
    # Initializing
    ###########################################################################
    def __init__(self,batch_size, gamma, size_replaybuffer,
                 optim_learning_rate):
        """
        Setting up the needed parameters for the HDRLN module.
        Setting up:
        - DeepSkillNetwork
        - Controller
        - Skill Experience Replay Buffer
        - StudentNetwork (distilled Neural Network)
        """
        # TODO (NOT): ACTION SPACE DEFINED A PRIORI--> Primitive action skill module???
        self.DSN_type = 'array'
        # self.device = self._setup_torch()
        self.batch_size = batch_size
        self.gamma = gamma
        self.size_replaybuffer = size_replaybuffer
        self.optim_learning_rate = optim_learning_rate

        self.loss = 0
        self.update_cnt = 0  # Student network update_counter

        self.DSN = DeepSkillNetwork()
        self.Controller = Controller()
        self.memory = SERB(1000)
        self.Student = StudentNetwork()

    def add(self, skill_dir, skill_name_list, agent_list):
        """
        Wrapper for the DSN object of the HDRLN agent.
        Returns the number of registered skills of the agent.
        """
        self.N_skills = self.DSN.add(skill_dir, skill_name_list, agent_list)

    def seal_skill_space(self):
        self.N_skills = self.DSN.seal()
        self.Controller.seal_skills(self.N_skills)

    ###########################################################################
    # Policy
    ###########################################################################

    def policy(self, state, available_actions, mode):
        """
        """
        self.mode = mode
        self.Controller.set_available_actions(available_actions)
        self.skill_idx = self.skill_policy(state) # Choose a skill
        print("skill index:" + str(self.skill_idx))
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
        if self.DSN_type is 'array':
            self.action_idx = self.DSN.pick(state, skill_idx)
        elif self.DSN_type is 'distilled':
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

    def update_student_network(self):
        """
        """
        self.Student.update()
