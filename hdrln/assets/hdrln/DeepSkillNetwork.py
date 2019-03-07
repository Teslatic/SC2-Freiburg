
class DeepSkillNetwork():
	""" DeepSkillNetwork as in described in lifelong learning paper """
	def __init__(self):
		self.N_skills = 0
		self.skills = []
		self.sealed = False

	def __len__(self):
		return len(self.skills)

	def __getitem__(self, idx):
		return self.skills[idx]

	###########################################################################
	# Adding skills
	###########################################################################

	def add(self,skill_dir, skill_name_list, agent_list):
		"""

		"""
		if not self.sealed:
			self.agent_list = self.load_models(skill_dir, skill_name_list, agent_list)
			self.skill_list = self.extract_skills(self.agent_list)
			self.add_skill_list(self.skill_list)
			self.N_skills = self.__len__()
			return self.N_skills
		else:
			print("Cannot add skills, cause DSN array is sealed!")
			exit()
			return self.N_skills

	def seal(self):
		"""
		"""
		self.sealed = True
		return self.N_skills

	def load_models(self,skill_dir, skill_name_list, agent_list):
		"""
		Loads the skills (models) at the skill directory into the agents in the agent list.
		"""
		N_agent = len(agent_list)
		N_skills = len(skill_name_list)
		if N_agent != N_skills:
			raise NameError("Number of agents not equal to number of skills!")
		for idx in range(N_agent):
			skill_name = skill_name_list[idx]
			# Load model into agent
			agent_list[idx].load_model(skill_dir + '/' + skill_name + '/model.pt')
		return agent_list

	def extract_skills(self,agent_list):
		"""
		Loads the skills (models) at the skill directory into the agents in the agent list.
		"""
		skill_list =[]
		for idx in range(len(agent_list)):
			skill_list.append(agent_list[idx].extract_skill()) # Extract DQN skill network
		return skill_list

	def add_skill_list(self, skill_specs_list):
		"""
		"""
		for skill in skill_specs_list:
			self.add_skill(skill)

	def add_skill(self, SkillObject):
		"""
		Adds a DQN module object as a skill to the DeepSkillNetwork.
		Adds one SkillObject to the list of skills to learn.
		Returns the number of added skills.
		"""
		# %TODO(vloet): check if skill admissible. Output layer needs to be softmax.
		self.skills.append(SkillObject)



	def prepare_skill_selection(self, obs):
		"""
		"""
		# state = ...
		state_start = state
		reward_cum = 0
		k = 0

	def pick(self, state, idx):
		"""
		Calculates the output of the skill at index position idx.
		"""
		return self.skills[idx].pick_best(state)
