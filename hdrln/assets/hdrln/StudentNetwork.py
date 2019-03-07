from assets.RL.DQN_module import DQN_module


class StudentNetwork(DQN_module):
	""" Student network that learns from the teacher networks """
    ###########################################################################
    # Initializing
    ###########################################################################

	# def __init__(self, architecture_file, N_skills):
	# 	self.N_skills = N_skills
	# 	batch_size = architecture_file.batch_size
	# 	gamma = architecture_file.gamma
	# 	history_length = architecture_file.history_length
	# 	size_replaybuffer = architecture_file.size_replaybuffer
	# 	optim_learning_rate = architecture_file.optim_learning_rate
	# 	# Output dimension is N_primitive = N_skills
	# 	dim_actions = architecture_file.dim_actions

	def __init__(self):

		self.batch_size = 32
		self.gamma = 0.9
		self.history_length = 1
		self.size_replaybuffer = 1000
		self.optim_learning_rate = 0.001
		# Output dimension is N_primitive = N_skills
		self.dim_actions = 400
		module_specs = self.zip_module_specs()
		super(StudentNetwork, self).__init__(module_specs)

	def zip_module_specs(self):
		"""
		"""
		module_specs = {
            'BATCH_SIZE': self.batch_size,
            'GAMMA': self.gamma,
            'HISTORY_LENGTH': self.history_length,
            'SIZE_REPLAYBUFFER': self.size_replaybuffer,
            'OPTIM_LR': self.optim_learning_rate,
            'DIM_ACTIONS': self.dim_actions
			}
		return module_specs

	def predict(self, state):
		with torch.no_grad():
			state_tensor = torch.tensor([state],
	                                    device=self.device,
	                                    dtype=torch.float,
	                                    requires_grad=False)
		return self.net(state_tensor)

	def td_target(self, k):
		"""
		Overriding DQN_module method according to paper.
        Calculating the TD-target for given q-values.
        gamma has to be multiplied according to to taken steps.
        """
		return (q_values_best_next * self.gamma**k) + self.reward_batch

	def update(self):
	    """
	    """
	    pass
