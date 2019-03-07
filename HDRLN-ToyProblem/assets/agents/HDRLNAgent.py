from Controller import Controller
from DeepSkillNetwork import DeepSkillNetwork
from SkillExperienceReplayBuffer import SERB
from StudentNetwork	import StudentNetwork

from assets.helperFunctions.timestamps import print_timestamp as print_ts

class HDRLNAgent():
	"""
	HDRLN framework as described in the long life learning with minecraft
	paper. The framework consists of 4 different parts:
	1. A DeepSkillNetwork which holds the networks for the teacher skills.
	2. A controller that can select between skills and primitive actions.
	3. A student network that learns from the teacher skills.
	4. A SkillExperienceReplayBuffer that is a modified version of a normal
	experience replay buffer.
	"""
	def __init__(self, agent_specs, N_skills):
		self.DSN = DeepSkillNetwork()
		self.Controller = Controller()
		self.SERB = SERB(1000)
		self.Student = StudentNetwork(N_skills)

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
