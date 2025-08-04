import numpy as np
import zipfile
import json
import sys
import pickle
import pandas as pd

# sys.path.append('../')
# import myconfig


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


model_path = '/home/ivan/PycharmProjects/Deep_CA1/outputs/big_models/base_model.keras'
target_path =  '/home/ivan/PycharmProjects/Deep_CA1/outputs/'

generator_params = get_gen_params(model_path)
net_params = get_net_params(model_path)

populations = pd.read_excel('../parameters/neurons_parameters.xlsx', sheet_name='theta_model')
populations.rename( {'neurons' : 'type'}, axis=1, inplace=True)
populations = populations[populations['Npops'] > 0]

pop_types = populations['type'].to_list()

data = {
    'generator_params' : generator_params,
    'net_params' : net_params,
}
# Сериализация списка в файл
with open(target_path + 'params.pickle', 'wb') as pfile:
    pickle.dump(data, pfile)


pconn_mask = net_params['pconn'] == 0
Ngenerators = len(generator_params)


writer = pd.ExcelWriter(target_path + 'params.xlsx')

for key, val in net_params.items():

    print(key, val.shape)

    if len(val.shape) == 1:
        df = pd.DataFrame(val, columns=[key, ]) #, columns=pop_types[:-Ngenerators])
        df.insert(0, 'Neuron type', pop_types[:-Ngenerators])
    else:
        val[pconn_mask] = np.nan
        df = pd.DataFrame(val, columns=pop_types[:-Ngenerators])
        df.insert(0, 'Presynaptic', pop_types)

    df.to_excel(writer, sheet_name=key)

writer.close()


