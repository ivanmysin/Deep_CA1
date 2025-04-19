import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import h5py
import izhs_lib
from pprint import pprint


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RNN, Reshape
from mean_field_class import MeanFieldNetwork
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import os
os.chdir('../')
#from genloss import SpatialThetaGenerators, CommonOutProcessing, PhaseLockingOutputWithPhase, PhaseLockingOutput, RobastMeanOut, FiringsMeanOutRanger, Decorrelator
import myconfig


def get_params_from_pop_conns(populations, connections, neurons_params, synapses_params, dt_dim, Delta_eta):

    params = {}
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



    NN = len(populations) - len(generators_params)
    Ninps = len(generators_params)
    gsyn_max = np.zeros(shape=(NN + Ninps, NN), dtype=np.float32)

    dimpopparams['gsyn_max'] = gsyn_max
    dimpopparams["Erev"] = np.zeros_like(gsyn_max)

    params['pconn'] = np.zeros_like(gsyn_max)
    params['tau_d'] = np.zeros_like(gsyn_max) #+ tau_d
    params['tau_r'] = np.zeros_like(gsyn_max) #+ tau_r
    params['tau_f'] = np.zeros_like(gsyn_max) #+ tau_f
    params['Uinc'] = np.zeros_like(gsyn_max) #+ Uinc

    for conn in connections:
        pre_idx = conn['pre_idx']
        post_idx = conn['post_idx']

        params['pconn'][pre_idx, post_idx] = conn['pconn']

        pre_type = conn['pre_type']

        if pre_type == "CA3_generator":
            pre_type = 'CA3 Pyramidal'

        if pre_type == "CA1 Pyramidal_generator":
            pre_type = 'CA1 Pyramidal'

        if pre_type == "MEC_generator":
            pre_type = 'EC LIII Pyramidal'

        if pre_type == "LEC_generator":
            pre_type = 'EC LIII Pyramidal'

            # pre_type = pre_type.replace("_generator", "")

        syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == pre_type) & (
                synapses_params['Postsynaptic Neuron Type'] == conn['post_type'])]

        if len(syn) == 0:
            print("Connection from ", conn["pre_type"], "to", conn["post_type"], "not finded!")
            continue

        Uinc = syn['Uinc'].values[0]
        tau_r = syn['tau_r'].values[0]
        tau_f = syn['tau_f'].values[0]
        tau_d = syn['tau_d'].values[0]

        if neurons_params[neurons_params['Neuron Type'] == pre_type]['E/I'].values[0] == "e":
            Erev = 0
        elif neurons_params[neurons_params['Neuron Type'] == pre_type]['E/I'].values[0] == "i":
            Erev = -75.0

        params['Uinc'][pre_idx, post_idx] = Uinc
        params['tau_r'][pre_idx, post_idx] = tau_r
        params['tau_f'][pre_idx, post_idx] = tau_f
        params['tau_d'][pre_idx, post_idx] = tau_d
        dimpopparams['Erev'][pre_idx, post_idx] = Erev

    params_dimless = izhs_lib.dimensional_to_dimensionless_all(dimpopparams)


    params = params | params_dimless
    params['I_ext'] = np.zeros(NN, dtype=np.float32)
    print(params.keys())

    return params




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

net_layer = RNN( MeanFieldNetwork(params, dt_dim=dt_dim, use_input=True), return_sequences=True, stateful=True)
