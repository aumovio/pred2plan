from distutils.dir_util import copy_tree
import os
import shutil


def copy_files(log_dir):
    if not os.path.exists(log_dir + "/scripts/"):
        os.makedirs(log_dir + "/scripts/")
        
    shutil.copytree('datamodules', log_dir + "/scripts/datamodules", dirs_exist_ok=True)  # Fine
    shutil.copytree('datasets', log_dir + "/scripts/datasets", dirs_exist_ok=True)  # Fine
    shutil.copytree('losses', log_dir + "/scripts/losses", dirs_exist_ok=True)  # Fine
    shutil.copytree('layers', log_dir + "/scripts/layers", dirs_exist_ok=True)  # Fine
    shutil.copytree('modules', log_dir + "/scripts/modules", dirs_exist_ok=True)  # Fine
    shutil.copytree('predictors', log_dir + "/scripts/predictors", dirs_exist_ok=True)  # Fine
    shutil.copy("train_ep.py", log_dir + "/scripts/")