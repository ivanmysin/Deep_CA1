import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import h5py
import izhs_lib
from pprint import pprint


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RNN, Reshape
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import os
os.chdir('../')
#from genloss import SpatialThetaGenerators, CommonOutProcessing, PhaseLockingOutputWithPhase, PhaseLockingOutput, RobastMeanOut, FiringsMeanOutRanger, Decorrelator
import myconfig


def get_params_from_pop_conns(populations, connections, neurons_params, synapses_params, dt_dim, Delta_eta):

    dimpopparams = {
        'dt_dim' : dt_dim,
        'Delta_eta' : Delta_eta,
    }
    generators_params = []

    for pop in populations:
        pop_type = pop['type']

        if 'generator' in pop_type:
            generators_params.append(pop)
            continue

        p = neurons_params[neurons_params["Neuron Type"] == pop_type]

        for key in p:
            val = p[key].values[0]
            try:
                val  = float(val)
            except ValueError:
                continue

            if key in dimpopparams.keys():
                dimpopparams[key].append(val)
            else:
                dimpopparams[key] = [val, ]

    for key, val in dimpopparams.items():
        dimpopparams[key] = np.asarray(val)

    params = izhs_lib.dimensional_to_dimensionless_all(dimpopparams)


    for conn in connections:
        pre_idx = conn['pre_idx']
        post_idx = conn['post_idx']

        print(conn.keys())

        break
    print(params.keys())




##################################################################
dt_dim = 0.5
Delta_eta = 80



# load data about network
if myconfig.RUNMODE == 'DEBUG':
    neurons_path = myconfig.STRUCTURESOFNET + "test_neurons.pickle"
    connections_path = myconfig.STRUCTURESOFNET + "test_conns.pickle"
else:
    neurons_path = myconfig.STRUCTURESOFNET + "neurons.pickle"
    connections_path = myconfig.STRUCTURESOFNET + "connections.pickle"

with open(neurons_path, "rb") as neurons_file:
    populations = pickle.load(neurons_file)

with open(connections_path, "rb") as synapses_file:
    connections = pickle.load(synapses_file)

neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
neurons_params.rename(
        {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)


params = get_params_from_pop_conns(populations, connections, neurons_params, synapses_params, dt_dim, Delta_eta)