import numpy as np
import zipfile
import json
import sys

from Cython.Shadow import returns

sys.path.append('../')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RNN, Layer
from mean_field_class import MeanFieldNetwork, SaveFirings
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN


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

input = Input(shape=(None, 1), batch_size=1)

generators = SpatialThetaGenerators(generators_params)(input)
net_layer = RNN(MeanFieldNetwork(params, dt_dim=myconfig.DT, use_input=True),
                    return_sequences=True, stateful=True, return_state=True,
                    name="firings_outputs")(generators)
outputs = net_layer # generators #
big_model = Model(inputs=input, outputs=outputs)

t = np.arange(0, 10, myconfig.DT, dtype=np.float32).reshape(1, -1, 1)

states = big_model.predict(t)

for s in states:
    print(s.shape)
