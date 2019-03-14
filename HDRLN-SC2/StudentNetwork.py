from assets.RL.DQN_module import DQN_module

class StudentNetwork(DQN_module):
	""" Student network that learns from the teacher networks """

	# Student Network should not have Softmax output ???
	# Probably not an issue
	def __init__(self, architecture_file, N_skills):
		self.N_skills = N_skills
		batch_size = architecture_file.batch_size
		gamma = architecture_file.gamma
		history_length = architecture_file.history_length
		size_replaybuffer = architecture_file.size_replaybuffer
		optim_learning_rate = architecture_file.optim_learning_rate
		# Output dimension is N_primitive = N_skills
		dim_actions = architecture_file.dim_actions

		super(StudentNetwork, self).__init__(batch_size, gamma, history_length, size_replaybuffer,
                 optim_learning_rate, dim_actions)


	def td_target(self, k):
		"""
		Overriding DQN_module method according to paper.
        Calculating the TD-target for given q-values.
        gamma has to be multiplied according to to taken steps.
        """
		return (q_values_best_next * self.gamma**k) + self.reward_batch
