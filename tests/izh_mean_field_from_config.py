import numpy as np
import zipfile
import json
import sys

import matplotlib.pyplot as plt

sys.path.append('../')

from np_meanfield import MeanFieldNetwork

from genloss import SpatialThetaGenerators
import myconfig

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


model_path = '/home/ivan/PycharmProjects/Deep_CA1/outputs/big_models/big_model_20.keras'

params = get_net_params(model_path)
generators_params = get_gen_params(model_path)

generators = SpatialThetaGenerators(generators_params)
t = np.arange(0, 400, myconfig.DT, dtype=np.float32).reshape(1, -1, 1)

generators_firings = generators(t)
generators_firings = generators_firings.numpy()

print(generators_firings.shape)
model = MeanFieldNetwork(params, dt_dim=myconfig.DT, use_input=True)


firing, states = model.predict(generators_firings)
firing = firing.reshape(-1, firing.shape[-1])
generators_firings = generators_firings.reshape(-1, generators_firings.shape[-1])
t = t.ravel()

plt.plot(t, generators_firings)
plt.show()



