
class DeepSkillNetwork():
	""" DeepSkillNetwork as in described in lifelong learning paper """
	def __init__(self):
		# self.num_skills = 0
		self.skills = []

	def __len__(self):
		return len(self.skills)

	def __getitem__(self, idx):
		return self.skills[idx]

	def add_skill(self, SkillObject):
		"""
		Adds one SkillObject to the list of skills to learn.
		Returns the number of added skills.
		"""
		# %TODO(vloet): check if skill admissible. Output layer needs to be softmax.
		self.skills.append(SkillObject)
		return self.__len__()
		# self.update_skill_array_length()


	# def update_skill_array_length(self):
	# 	self.num_skills = len(self.skills)

	def forward(self, idx, state):
		"""
		Calculates the output of the skill at index position idx.
		"""
		return self.skills[idx].forward(state)

