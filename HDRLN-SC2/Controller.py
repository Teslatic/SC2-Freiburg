
class Controller():

    def skill_policy(self, obs, reward, done, info):
        """
        Choosing a skill or primitive action.
        """
        # Set all variables at the start of a new timestep
        self.prepare_timestep(obs, reward, done, info)

        # Skill selection for regular step
        if not self.last:
            if self.mode == 'supervised':
                # For the first n episodes learn on forced skill.
                self.skill, self.skill_idx = self.supervised_skill()
            if self.mode == 'learning':
                # Choose an skill according to the policy
                self.skill, self.skill_idx = self.epsilon_greedy_skill()
            if self.mode == 'testing':
                self.skill, self.skill_idx = self.pick_skill()
            else:
                self.skill, self.skill_idx = 'no skill', 'no skill_idx'

        if self.last:  # End episode in last step
            self.skill = 'reset'
            if self.mode != 'testing':
                self.update_target_network()
            self.reset()

        return self.skill, self.skill_idx