# python imports
import csv
import os
from os import path
import sys
import pandas as pd
from assets.helperFunctions.timestamps import print_timestamp as print_ts
from assets.helperFunctions.FileManager import FileManager

class HDRLNFileManager(FileManager):
    """
    The FileManager class is used to store and access experiment data with
    ease and without the hazard of losing data.
    """
    def extract_skill_specs(self, skill_name_list):
        """
        Import learned agent networks
        """
        skill_list = []
        for skill_name in skill_name_list:
            skill_specs_path = self.main_dir + "/skills/" + skill_name + "/specs.csv"
            skill_model_path = self.main_dir + "/skills/" + skill_name + "/model.pt"
            skill_specs = self.load_spec_summary(skill_specs_path)
            # model =
            print(skill_specs)
            skill_list.append(skill_specs)
        return skill_list
