# python imports
import numpy as np

# custom imports
from assets.agents.DQNBaseAgent import DQNBaseAgent
from assets.hdrln.HDRLN_module import HDRLN_module
from assets.helperFunctions.timestamps import print_timestamp as print_ts
from assets.actionspace.utils import setup_action_space

class HDRLNAgent(DQNBaseAgent):
	"""
	HDRLN framework as described in the long life learning with minecraft
	paper. The framework consists of 4 different parts:
	1. A DeepSkillNetwork which holds the networks for the teacher skills.
	2. A controller that can select between skills and primitive actions.
	3. A student network that learns from the teacher skills.
	4. A SkillExperienceReplayBuffer that is a modified version of a normal
	experience replay buffer.
	"""
    ###########################################################################
    # Initializing
    ###########################################################################

	def __init__(self, agent_specs):
		"""
		1. Unzipping the agent specs
		2. Setup additional framework parameters
		3. Setup DSN, Controller, SERB and Student Network
		"""
		super(HDRLNAgent, self)._setup_base_agent()
		super(HDRLNAgent, self).unzip_agent_specs(agent_specs)
		super(HDRLNAgent, self)._setup_framework_parameters()

		self.action_space, self.dim_actions = setup_action_space(agent_specs)
		self._setup_hdrln()


	def _setup_hdrln(self):
		"""
		Setting up the HDRLN module.
		"""
		self.HDRLN = HDRLN_module(batch_size=self.batch_size,
							gamma=self.gamma,
							size_replaybuffer=self.size_replaybuffer,
							optim_learning_rate=self.optim_learning_rate)
		print_ts("HDRLN module has been initalized")

	def add_skills(self, skill_dir, skill_name_list, agent_list):
		"""
		Returns the number of registered skills of the agent.
		"""
		self.N_skills = self.HDRLN.add(skill_dir, skill_name_list, agent_list)

	def seal_skill_space(self):
		"""
		"""
		self.HDRLN.seal_skill_space()

	###########################################################################
    # Policy
    ###########################################################################

	def policy(self, obs, reward, done, info):
		"""
		The policy is mainly carried out by the Controller.
		1. Prepare timestep
		2. Mode switch
		3.
		"""
		# Set all variables at the start of a new timestep
		self.prepare_timestep(obs, reward, done, info)

		# Skill selection depending on agent mode
		if not self.last:
			self.action_idx = self.HDRLN.policy(state=self.state,
									   available_actions=self.available_actions,
									   mode=self.mode)
			# Select action from actoin space with action_idx
			self.action = self.action_space[self.action_idx]

		if self.last:
			self.action = 'reset'
			if self.mode is 'learning':
				self.HDRLN.update_student_network() #
				self.reset()
		return self.action

	def prepare_timestep(self, obs, reward, done, info):
		"""
		Prepares the timestep:
		1. Extracting obs and reward information as from DQNBaseAgent
		"""
		super(HDRLNAgent, self).prepare_timestep(obs, reward, done, info)

	def reset(self):
		"""
		Resetting the agent
		"""
		super(HDRLNAgent, self).reset()

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

		self.HDRLN.memory.push([self.state],
                             [self.action_idx],
                             self.reward,
                             [self.next_state])

	def optimize(self):
		"""
		Optimizes the DQN_module on a minibatch
		"""
		if self.get_memory_length() >= self.batch_size:
			self.loss = self.HDRLN.optimize()

	def get_memory_length(self):
		"""
		Returns the length of the ReplayBuffer
		"""
		return len(self.HDRLN.memory)

	def save_model(self, exp_path, emergency=False):
		"""
		"""
		pass
