import os
import json


def save_config(config_path):
    with open(config_path, 'w') as fp:
        json.dump(config, fp, indent=4)
        
def load_config(config_path):
    with open(config_path) as fp:
        config = json.load(fp)
    return config

def get_last_file(path):
    files = [x for x in os.listdir(path)]
    newest = max([path + "/" + i for i in files], key = os.path.getctime)
    return newest 

def get_sorted_files(path):
    files = [x for x in os.listdir(path)]
    sorted_files = sorted([path + "/" + i for i in files], key = os.path.getctime)
    return sorted_files