import numpy as np
import zipfile
import json

def get_net_params(filepath):
    with zipfile.ZipFile(filepath, mode='r') as zipped_file:

        config_file = zipped_file.open("config.json", mode='r')
        config = json.loads(config_file.read().decode())

        config_file.close()

    # gen_params = config['config']['layers'][1]['config']['myparams']
    netconfig = config['config']['layers'][2]['config']['cell']['config']
    params = {}
    for key, vals in netconfig.items():

        try:
            v = vals['config']['value']

        except KeyError:
            continue

        except TypeError:
            continue

        params[key] = np.asarray(v).astype(np.float32)

    return params

def get_gen_params(filepath):
    with zipfile.ZipFile(filepath, mode='r') as zipped_file:

        config_file = zipped_file.open("config.json", mode='r')
        config = json.loads(config_file.read().decode())

        config_file.close()

    config_gen_params = config['config']['layers'][1]['config']['myparams']
    return config_gen_params