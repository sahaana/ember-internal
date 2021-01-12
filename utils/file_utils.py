import json


def save_config(config_path):
    with open(config_path, 'w') as fp:
        json.dump(config, fp, indent=4)
        
def load_config(config_path):
    with open(config_path) as fp:
        config = json.load(fp)
    return config