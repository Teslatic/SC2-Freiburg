# q_max, q_min, q_mean, q_var, q_span, q_argmax = self.q_value_analysis(self.state_q_values)
# td_max, td_min, td_mean, td_var, td_span, td_argmax = self.q_value_analysis(self.td_target.unsqueeze(1))
# # print("environment reward: {:.2f}, step penalty: {:.2f}, reward total: {:.2f}".format(obs.reward, self.reward_shaped, self.reward_combined))
# # print("Q_VALUES: {}".format(self.state_q_values))
# # print("TD_TARGET: {}".format(self.td_target.unsqueeze(1)))
# print_ts("action: {}, idx: {}, Smart action: {}".format(self.action, self.action_idx, SMART_ACTIONS[self.action_idx]))
# #print_ts("Q_VALUES: max: {:.3f}, min: {:.3f}, span: {:.3f}, mean: {:.3f}, var: {:.6f}, argmax: {}".format(q_max, q_min, q_span, q_mean, q_var, q_argmax))
# print_ts("TD_TARGET: max: {:.3f}, min: {:.3f}, span: {:.3f}, mean: {:.3f}, var: {:.6f}, argmax: {}".format(td_max, td_min, td_span, td_mean, td_var, td_argmax))
# print_ts("MEMORY SIZE: {}".format(len(DQN.memory)))
# print_ts("Epsilon: {:.2f}\t| choice: {}".format(self.epsilon, self.choice))
# print_ts("Episode {}\t| Step {}\t| Total Steps: {}".format(self.episodes, self.timesteps, self.steps))
# print_ts("Chosen action: {}".format(self.action))
# # print_ts("chosen coordinates [x,y]: {}".format((self.x_coord.item(), self.y_coord.item())))
# print_ts("Beacon center location [x,y]: {}".format(self.beacon_center))
# print_ts("Marine center location [x,y]: {}".format(self.marine_center))
# print_ts("Distance: {}, delta distance: {}".format(self.distance, self.reward_shaped))
# print_ts("Current Episode Score: {}\t| Total Score: {}".format(self.last_score, self.reward))
# print_ts("Environment reward in timestep: {}".format(self.env_reward))
# print_ts("Action Loss: {:.5f}".format(self.loss))
# print_ts("----------------------------------------------------------------")

    ## logs rewards per epochs and cumulative reward in a csv file
    # def log_reward(self):
    #     r_per_epoch_save_path = self.exp_path  + "/csv/reward_per_epoch.csv"
    #     # Path((self._path + "/experiment/" + self._name + "/csv")).mkdir(parents=True, exist_ok=True)
    #
    #     d = {"sparse reward per epoch" : self.r_per_epoch,
    #          "sparse cumulative reward": self.list_score_cumulative,
    #          "pseudo reward per epoch" : self.list_pseudo_reward_per_epoch,
    #          "epsilon" : self.list_epsilon }
    #     df = pd.DataFrame(data=d)
    #     with open(r_per_epoch_save_path, "w") as f:
    #         df.to_csv(f, header=True, index=False)
    #
    #
    # def log_coordinates(self):
    #     save_path = self.exp_path + "/csv/coordinates.csv"
    #     # Path((self._path + "/experiment/" + self._name + "/csv")).mkdir(parents=True, exist_ok=True)
    #     d = {"x" : self.list_x,
    #          "y":  self.list_y}
    #     df = pd.DataFrame(data=d)
    #     with open(save_path, "w") as f:
    #         df.to_csv(f, header=True, index=False)


    def print_status(self):
        """
        Method for direct status printing.
        The method is called at the end of each timestep when optimizing is
        active.
        """
        # print(self.state_q_values_full)
        if self.episodes < self.supervised_episodes:
            pass
        else:
            print(self.feature_screen)

    def log(self):
        pass
        buffer_size = 10  # This makes it so changes appear without buffering
        with open('output.log', 'w', buffer_size) as f:
                f.write('{}\n'.format(self.feature_screen))
