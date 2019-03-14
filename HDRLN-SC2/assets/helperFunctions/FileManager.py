# python imports
import csv
import os
from os import path
import sys
import pandas as pd
from assets.helperFunctions.timestamps import print_timestamp as print_ts


class FileManager():
    """
    The FileManager class is used to store and access experiment data with
    ease and without the hazard of losing data.
    """
    def __init__(self):
        self.main_dir = path.dirname(path.abspath(sys.modules['__main__'].__file__))
        self.exp_dir = '{}/{}'.format(self.main_dir, 'experiments')

        # current directories
        self.cwd = None
        self.current_test_dir = None
        self.current_train_dir = None

    def change_cwd(self, change_path):
        """
        Change cwd if directory exists.
        """
        if not path.exists(change_path):
            print_ts("Directory is unknown")
            return
        self.cwd = change_path
        self.current_test_dir = change_path + "/test_reports"
        self.current_train_dir = change_path + "/training_reports"
        print_ts("Changed cwd to {}".format(self.cwd))

    def create_experiment(self, exp_name):
        """
        Creates the folder structure which is necessary to save files.
        """
        # Create free path
        free_exp_path, _ = self.find_free_path(self.exp_dir, exp_name)
        # print(free_exp_path, exp_name)
        # Create the experiment folders
        creation_success = self.create_folders(free_exp_path)
        if creation_success:
            print_ts("Created experiment at path {}".format(free_exp_path))
            self.change_cwd(free_exp_path)
        else:
            pass

    def print_cwd(self):
        """
        """
        print_ts(self.cwd)

    def get_cwd(self):
        """
        """
        return self.cwd

    def create_folders(self, exp_path):
        """
        """
        try:
            # print_timestamp('Created  at path {}'.format(exp_path))
            os.makedirs(exp_path)
            # os.makedirs(exp_path+'/policy_plots')
            os.makedirs(exp_path+'/specs')
            os.makedirs(exp_path+'/training_reports')
            os.makedirs(exp_path+'/test_reports')
            os.makedirs(exp_path+'/model')
            os.makedirs(exp_path+'/plots/png')
            os.makedirs(exp_path+'/plots/pdf')
            return True
        except:
            return False

    def find_free_path(self, dir_path, exp_name):
        """
        Finds the next free index for the experiment name.
        """
        exp_directory = dir_path + '/' + exp_name
        file_index = 1
        while path.exists(exp_directory + '%s' % file_index):
            file_index += 1
        return exp_directory + '{}'.format(file_index), exp_name + '{}'.format(file_index)


    # Saving and Loading

    def save_specs(self, agent_specs, env_specs):
        """
        Saves agent and environment specs as csv
        """
        spec_path = self.cwd + "/specs"
        exp_path_dict = {'ROOT_DIR': self.cwd}
        spec_dict = {**exp_path_dict, **agent_specs, **env_specs}
        spec_df = pd.DataFrame.from_dict(spec_dict, orient="index")
        with open(spec_path + "/spec_summary.csv", "w") as f:
            spec_df.to_csv(f, header=False, index=True)

    def load_spec_summary(self, spec_path):
        """
        Loads the spec summary into a pandas DataFrame
        """
        with open(spec_path, mode='r') as infile:
            reader = csv.reader(infile)
            spec_summary = {rows[0]:rows[1] for rows in reader}
        return spec_summary

    def log_training_reports(self, training_report):
        """
        Merges report dictionaries and saves them as csv
        """
        if training_report is not None:
            dictb = {"MergeDummy": 0}
            dict_complete = {**training_report, **dictb}
            logging_df = pd.DataFrame.from_dict(dict_complete)

            with open(self.current_train_dir + "/training_report.csv", "w") as f:
                logging_df.to_csv(f, header=True, index=False)

    def create_test_file(self):
        self.current_testfile = self.create_free_name_test('test_report')

    def create_train_file(self):
        self.current_trainfile = self.create_free_name_train('training_report')

    def log_test_reports(self, test_report):
        """
        Merges report dictionaries and saves them as csv
        """
        if test_report is not None:
            dictb = {"MergeDummy": 0}
            dict_complete = {**test_report, **dictb}
            logging_df = pd.DataFrame.from_dict(dict_complete)

            with open(self.current_test_dir + "/" + self.current_testfile + ".csv", "w") as f:
                logging_df.to_csv(f, header=True, index=False)

    def create_free_name_train(self, exp_name):
        exp_directory = self.current_train_dir + '/' + exp_name
        file_index = 1
        while path.isfile(exp_directory + '%s' % file_index + '.csv'):
            file_index += 1
        return exp_name + '{}'.format(file_index)

    def create_free_name_test(self, exp_name):
        exp_directory = self.current_test_dir + '/' + exp_name
        file_index = 1
        while path.isfile(exp_directory + '%s' % file_index + '.csv'):
            file_index += 1
        return exp_name + '{}'.format(file_index)

    def extract_skill_specs(self, skill_name_list):
        """
        Import learned agent networks
        """
        skill_list = []
        for skill_name in skill_name_list:
            skill_specs_path = self.main_dir + "/assets/skills/" + skill_name + "/specs.csv"
            skill_model_path = self.main_dir + "/assets/skills/" + skill_name + "/model.pt"
            skill_specs = self.load_spec_summary(skill_specs_path)
            # model =
            print(skill_specs)
            skill_list.append(skill_specs)
        return skill_list
