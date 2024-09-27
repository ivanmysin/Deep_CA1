import os

from keras.src.ops import dtype

os.chdir("../")
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import myconfig
from synapses_layers import TsodycsMarkramSynapse


pop_params = pd.read_excel( myconfig.SCRIPTS4PARAMSGENERATION + "neurons_parameters.xlsx", sheet_name="Sheet2", header=0)

neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
synapses_params.rename({"g" : "gsyn_max", "u" : "Uinc", "Connection Probability":"pconn"}, axis=1, inplace=True)

pop_types_models = {}

for pop_idx, pop_type in pop_params.iterrows():
    pop_name = pop_type["neurons"]

    #!!!!!!!!! modelfile = myconfig.PRETRANEDMODELS + pop_type["neurons"] + ".keras"
    modelfile = myconfig.PRETRANEDMODELS + "pv_bas" + ".keras"

    try:
        pop_types_models[pop_name] = tf.keras.models.load_model(modelfile)
    except ValueError:
        print(f"File for model population {pop_name} is not found")


with open(myconfig.STRUCTURESOFNET + "neurons.pickle", "rb") as neurons_file:
    neurons_populations_params = pickle.load(neurons_file)

with open(myconfig.STRUCTURESOFNET + "connections.pickle", "rb") as synapses_file:
    connections_params = pickle.load(synapses_file)


#print()

Npops = len(neurons_populations_params)
input_shape = (1, None, Npops)
pop_models = []
for pop_idx, pop in enumerate(neurons_populations_params):

    pop_type = pop["type"]

    conn_params = {
        "gsyn_max" : [], # np.zeros(Ns, dtype=np.float32) + 1.5,
        "Uinc" :  [], #np.zeros(Ns, dtype=np.float32) + 0.5,
        "tau_r" : [], #np.zeros(Ns, dtype=np.float32) + 1.5,
        "tau_f" : [], #np.zeros(Ns, dtype=np.float32) + 1.5,
        "tau_d" : [], #np.zeros(Ns, dtype=np.float32) + 1.5,
        'pconn' : [], #np.zeros(Ns, dtype=np.float32) + 1.0,
        'Erev' : [], #np.zeros(Ns, dtype=np.float32),
        'Cm' : neurons_params[neurons_params["Neuron Type"] == pop_type]["Izh C"].values[0],
        'Erev_min' : -75.0,
        'Erev_max' : 0.0,
    }

    is_connected_mask = np.zeros(Npops, dtype='bool')

    for conn in connections_params:
        if conn["pre_idx"] != pop_idx: continue

        syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == conn['pre_type']) & (
                    synapses_params['Postsynaptic Neuron Type'] == conn['post_type'])]

        if len(syn) == 0:
            continue

        is_connected_mask[conn["pre_idx"]] = True

        conn_params["gsyn_max"].append( np.random.rand()  ) # !!!
        conn_params['pconn'].append( conn['pconn']  ) # !!!




        Uinc = syn['Uinc'].values[0]
        tau_r = syn['tau_r'].values[0]
        tau_f = syn['tau_f'].values[0]
        tau_d = syn['tau_d'].values[0]

        if neurons_params[neurons_params['Neuron Type'] == conn['pre_type']]['E/I'].values[0] == "e":
            Erev = 0
        elif neurons_params[neurons_params['Neuron Type'] == conn['pre_type']]['E/I'].values[0] == "i":
            Erev = -75.0

        conn_params['Uinc'].append( Uinc ) # !!!
        conn_params['tau_r'].append( tau_r ) # !!!
        conn_params['tau_f'].append( tau_f ) # !!!
        conn_params['tau_d'].append( tau_d ) # !!!
        conn_params['Erev'].append( Erev ) # !!!



    synapses = TsodycsMarkramSynapse(conn_params, dt=myconfig.DT, mask=is_connected_mask)
    synapses_layer = tf.keras.layers.RNN(synapses, return_sequences=True, stateful=True)

    model = tf.keras.Sequential()
    model.add(synapses_layer)

    for layer in pop_types_models[pop_type].layers:
         model.add( tf.keras.models.clone_model(layer) )

    model.build(input_shape=input_shape)

    for l_idx, layer in enumerate(model.layers):
        if l_idx == 0:
            continue
        layer.trainable = False
        layer.set_weights(pop_types_models[pop_type].layers[l_idx-1].get_weights())


    #print(model.summary())

    break # !!!!!!!!!!!!!!!!!!!!


