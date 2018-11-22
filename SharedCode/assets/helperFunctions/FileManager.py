# python imports
import csv
import os
from os import path
import sys
import pandas as pd
from assets.helperFunctions.timestamps import print_timestamp as print_ts


def create_free_path(dir_path, exp_name):
    """
    Finds the next free index for the experiment name.
    """
    exp_directory = dir_path + '/' + exp_name
    file_index = 1
    while path.exists(exp_directory + '%s' % file_index):
        file_index += 1
    return exp_directory + '{}'.format(file_index)


def create_experiment(exp_path):
    """
    Creates the folder structure which is necessary to save files.
    """
    # print_timestamp('Created  at path {}'.format(exp_path))
    os.makedirs(exp_path)
    # os.makedirs(exp_path+'/policy_plots')
    os.makedirs(exp_path+'/specs')
    create_plots_dir(exp_path)
    create_report_dir(exp_path)
    create_model_dir(exp_path)


def create_report_dir(exp_path):
    os.makedirs(exp_path+'/report')


def create_model_dir(exp_path):
    os.makedirs(exp_path+'/model')


def create_plots_dir(exp_path):
    os.makedirs(exp_path+'/plots/png')
    os.makedirs(exp_path+'/plots/pdf')


def create_path_and_experiment(dir_path, exp_name):
    """
    Creates the path and the folder structure which is necessary to save files.
    """
    idx_exp_dir = create_free_path(dir_path, exp_name)
    create_experiment(idx_exp_dir)
    return idx_exp_dir


def create_experiment_at_main(exp_name):
    main_path = path.dirname(path.abspath(sys.modules['__main__'].__file__))
    all_exp_path = '{}/{}'.format(main_path, 'experiments')
    exp_root_dir = create_path_and_experiment(all_exp_path, exp_name)
    print_ts("Created eperiment at path {}".format(exp_root_dir))
    return exp_root_dir



def log_training_reports(agent_report, exp_path):
    """
    Merges report dictionaries and saves them as csv
    """
    report_path = exp_path + "/report"
    if agent_report is not None:
        dictb = {"MergeDummy": 0}
        dict_complete = {**agent_report, **dictb}
        logging_df = pd.DataFrame.from_dict(dict_complete)

        with open(report_path + "/training_report.csv", "w") as f:
            logging_df.to_csv(f, header=True, index=False)


def log_test_reports(test_report, exp_path):
    """
    Merges report dictionaries and saves them as csv
    """
    report_path = exp_path + "/report"
    if test_report is not None:
        dictb = {"MergeDummy": 0}
        dict_complete = {**test_report, **dictb}
        logging_df = pd.DataFrame.from_dict(dict_complete)

        with open(report_path + "/test_report.csv", "w") as f:
            logging_df.to_csv(f, header=True, index=False)




def save_specs(agent_specs, env_specs, exp_path):
    """
    Saves agent and environment specs as csv
    """
    spec_path = exp_path + "/specs"
    exp_path_dict = {'ROOT_DIR': exp_path}
    spec_dict = {**exp_path_dict, **agent_specs, **env_specs}
    spec_df = pd.DataFrame.from_dict(spec_dict, orient="index")
    with open(spec_path + "/spec_summary.csv", "w") as f:
        spec_df.to_csv(f, header=False, index=True)


def load_spec_summary(spec_path):
    """
    Loads the spec summary into a pandas DataFrame
    """
    # spec_path = exp_path + "/specs"

    with open(spec_path, mode='r') as infile:
        reader = csv.reader(infile)
        spec_summary = {rows[0]:rows[1] for rows in reader}
    #
    # print(mydict)
    # exit()
    #
    # with open(spec_path, "r") as f:
    #     spec_summary = pd.read_csv(f, index_col=[0])
    return spec_summary
