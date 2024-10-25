import os
os.chdir("../")

import numpy as np
import tensorflow as tf
from keras import saving
import pandas as pd
import pickle

import myconfig
from synapses_layers import TsodycsMarkramSynapse
import genloss  #s import SpatialThetaGenerators
import net_lib


with open(myconfig.STRUCTURESOFNET + "test_neurons.pickle", "rb") as neurons_file:
    populations = pickle.load(neurons_file)

types = set([pop['type'] for pop in populations])
# print(types)

with open(myconfig.STRUCTURESOFNET + "test_conns.pickle", "rb") as synapses_file:
    connections = pickle.load(synapses_file)

pop_types_params = pd.read_excel(myconfig.SCRIPTS4PARAMSGENERATION + "neurons_parameters.xlsx", sheet_name="Sheet2",
                                 header=0)

neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
neurons_params.rename(
    {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
     'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)
synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)



net = net_lib.Net(populations, connections, pop_types_params, neurons_params, synapses_params)

for pop_idx, pop_layer in enumerate(net.pop_models):
    # if "_generator" in pop["type"]:
    #     continue

    mask = pop_layer.synapses.cell.mask.numpy()
    mask2 = np.zeros_like(mask)

    pconn =  pop_layer.synapses.cell.pconn.numpy()

    conn_indexes = []
    for conn in connections:
        if conn["post_idx"] == pop_idx:
            #conn_indexes.append(conn["pre_idx"])

            mask2[conn["pre_idx"]] = True

    #conn_indexes = np.asarray(conn_indexes)


    #mask2[conn_indexes] = True
    print("##############################################")
    print(mask2)

    print(mask)
    print(np.sum(  (mask != mask2) ) )

    print(pconn.size, np.sum(mask) )
    print(pconn )
