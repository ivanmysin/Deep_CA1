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
pop_models = []
for pop_idx, pop in enumerate(neurons_populations_params):
    #
    is_connected = np.zeros(Npops, dtype='bool')

    for conn in connections_params:
        if conn["pre_idx"] == pop_idx:
            is_connected[conn["pre_idx"]] = True



 # synapses = TsodycsMarkramSynapse(params, dt=dt, mask=mask)
 #
 #    synapses_layer = RNN(synapses, return_sequences=True, stateful=True)
 # model = tf.keras.Sequential()
 #    model.add(synapses_layer)
 #
 #    for layer in population_model.layers:
 #        model.add( tf.keras.models.clone_model(layer) )
 #
 #    model.build(input_shape=input_shape)
 #    for l_idx, layer in enumerate(model.layers[1:]):
 #        layer.trainable = False
 #
 #       layer.set_weights(population_model.layers[l_idx-1].get_weights())
 #
 #    model.build(input_shape=input_shape)
#model.layers[1]
# model.layers[2].set_weights(population_model.layers[1].get_weights())