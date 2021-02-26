import os
import json

from embedding_utils import param_header


def save_config(config_path):
    with open(config_path, 'w') as fp:
        json.dump(config, fp, indent=4)
        
def load_config(config_path):
    with open(config_path) as fp:
        config = json.load(fp)
    return config

def get_config_knn_dir(config_path, base_path="/lfs/1/sahaana/enrichment/ember/embedding"):
    conf = load_config(config_path)
    fpath = param_header(conf['batch_size'], 
                         conf['final_size'], 
                         conf['lr'], 
                         conf['pool_type'], 
                         conf['epochs'], 
                         conf['train_size'])
    path = f"{base_path}/models/{conf['model_name']}/{fpath}/knn" 
    return path

def get_last_file(path):
    files = [x for x in os.listdir(path)]
    newest = max([path + "/" + i for i in files], key = os.path.getctime)
    return newest 

def get_sorted_files(path):
    files = [x for x in os.listdir(path)]
    sorted_files = sorted([path + "/" + i for i in files], key = os.path.getctime)
    return sorted_files


def get_alpha_sorted_files(path):
    files = [x for x in os.listdir(path)]
    sorted_files = sorted([path + "/" + i for i in files])
    return sorted_files 