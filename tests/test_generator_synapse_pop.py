import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN, Input

from synapses_layers import TsodycsMarkramSynapse
import genloss
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tensorflow.keras.saving import load_model

params = [

    {
        "R": 0.4,
        "OutPlaceFiringRate": 0.5,
        "OutPlaceThetaPhase": 1.57,
        "InPlacePeakRate": 30.0,
        "CenterPlaceField": 2500.0,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0,  # np.deg2rad(10) * 10 * 0.001,
        "PrecessionOnset": -1.57,
        "ThetaFreq": 8.0,
    },

    {
        "R": 0.25,
        "OutPlaceFiringRate": 0.5,
        "OutPlaceThetaPhase": 3.14,
        "InPlacePeakRate": 8.0,
        "CenterPlaceField": 5000.0,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": np.deg2rad(10) * 10 * 0.001,
        "PrecessionOnset": -1.57,
        "ThetaFreq": 8.0,
    },


]
dt = 0.5

generators = genloss.SpatialThetaGenerators(params)


###############################################################
pre_types = ["CA1 Pyramidal", "CA3 Pyramidal"] #  "CA1 O-LM",
post_type = "CA1 Basket"
synparams = pd.read_csv("../parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv")
synparams.rename({"g" : "gsyn_max", "u" : "Uinc", "Connection Probability":"pconn"}, axis=1, inplace=True)

synparams['Erev'] = np.zeros( len(synparams), dtype=np.float64)
#print(synparams.keys())

neurons_params = pd.read_csv("../parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv", header=0, usecols=["Neuron Type", "E/I", "Izh C", "Izh Vr"])
neurons_params.rename({"Neuron Type" : "Presynaptic Neuron Type"}, axis=1, inplace=True)
#print(neurons_params.head(5))

selected_synparam =  synparams.loc[ (synparams["Presynaptic Neuron Type"].isin(pre_types)) & (synparams["Postsynaptic Neuron Type"] == post_type) ]

selected_synparam = selected_synparam.merge(neurons_params, how="left", on="Presynaptic Neuron Type", copy=True)




for idx, row in selected_synparam.iterrows():
    if row["E/I"].strip() == "i":
        selected_synparam.at[idx, 'Erev'] = -75.0
    else:
        selected_synparam.at[idx, 'Erev'] = 0.0
selected_synparam["pconn"] = 1.0

keys_of_arrs = ['pconn', 'Erev', 'gsyn_max', 'tau_f', 'tau_d', 'tau_r', 'Uinc']
synparam = {}
for key in keys_of_arrs:
    synparam[key] = np.asarray(selected_synparam[key])

#print(neurons_params[neurons_params["Presynaptic Neuron Type"] == post_type]["Izh C"])
synparam["Cm"] = float( neurons_params[neurons_params["Presynaptic Neuron Type"] == post_type]["Izh C"].values[0] )
synparam["Erev_min"] = -75.0
synparam["Erev_max"] = 0.0
synparam["Vrest"] = float( neurons_params[neurons_params["Presynaptic Neuron Type"] == post_type]["Izh Vr"].values[0] )
synparam["gsyn_max"][-2] *= 100.0
synparam["gsyn_max"][-1] *= 400.0
pprint(synparam)

input_shape = [1, None, 2]
conn_mask = np.ones(2, dtype='bool')
synapses_layer = RNN(TsodycsMarkramSynapse(synparam, dt=dt, mask=conn_mask), return_sequences=True, stateful=True)
population_model = load_model("../pretrained_models/CA1 Basket.keras", custom_objects={'square': tf.keras.ops.square})




t = tf.range(0, 10000.0, dt, dtype=tf.float64)
t = tf.reshape(t, shape=(1, -1, 1))
firings = generators(t) #* 0.001 * dt

Esynt = synapses_layer(firings)

pop_firings = population_model(Esynt)



# print(model.summary())
# print(model.trainable_weights)
#
# #########################################
# X = np.zeros([1, len(t), 2], dtype=np.float32)
# X[0, :, :] = firings.numpy() # !!!!
#
#
# Y = model.predict(X)

firings = firings.numpy()
firings = firings[0, :, :]
t = t.numpy().ravel()
Esynt = Esynt.numpy().ravel()

Esynt = Esynt*75 - 75
pop_firings = pop_firings.numpy().ravel()

fig, axes = plt.subplots(nrows=3, sharex=True)
axes[0].plot(t, firings)
axes[1].plot(t, Esynt)
axes[2].plot(t, pop_firings)
# axes[2].set_ylim(0, 80)

plt.show()