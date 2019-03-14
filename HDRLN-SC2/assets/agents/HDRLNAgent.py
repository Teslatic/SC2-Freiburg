from Controller import Controller
from DeepSkillNetwork import DeepSkillNetwork
from SkillExperienceReplayBuffer import SERB
from StudentNetwork	import StudentNetwork


class HDRLNAgent():
	"""
	HDRLN framework as described in the long life learning with minecraft
	paper """
	def __init__(self):

		self.Controller = Controller()
		self.DSN = DeepSkillNetwork()
		self.SERB = SERB()
		self.Student = StudentNetwork()

	def add_skill_list(self, skill_specs_list):
		"""
		"""
		for skill in skill_specs_list:
			self.add_skill(skill)

	def add_skill(self, skill):
		"""
		Adds a DQN module object as a skill to the DeepSkillNetwork.
		"""
		self.N_skills = self.DSN.add_skill(skill)

	def prepare_skill_selection(self, obs):
		"""
		"""
		# state = ...
		state_start = state
		reward_cum = 0
		k = 0

	def policy(self, obs, reward, done, info):
		# Skill selection if no skill active
		if not self.skill_active:
			self.prepare_skill_selection(obs)
			self.skill, self.skill_idx = self.Controller.skill_policy()
		self.DSN.forward(self.skill_idx, state)

	def save_model(self, exp_path, emergency=False):
		"""
		"""
		pass

	def store_transition(self, next_obs, reward):
		"""
		Save the actual information in the history. As soon as there has been
		enough data, the experience is sampled from the replay buffer.
		TODO: Maybe using uint8 for storing into ReplayBuffer
		"""
		# Don't store transition if first or last step
		if self.last:
			return

		self.reward = reward
		self.next_state = np.array(next_obs[0], dtype=np.uint8)

		self.DQN.memory.push([self.state],
                             [self.action_idx],
                             self.reward,
                             [self.next_state])

	def forward(self, idx, state):
		self.DeepSkillNetwork.forward(idx, state)
