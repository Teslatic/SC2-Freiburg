import os
from os import path
import sys
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
    os.makedirs(exp_path+'/parameters')
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
