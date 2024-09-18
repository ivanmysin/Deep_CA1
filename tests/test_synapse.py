import numpy as np
import tensorflow as tf
from keras.src.backend import shape
from keras.src.ops import dtype
from tensorflow.keras.layers import Layer, RNN

from synapses_layers import TsodycsMarkramSynapse
import genloss
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

params = [

    {
        "R": 0.4,
        "OutPlaceFiringRate": 30.0,
        "OutPlaceThetaPhase": 1.57,
        "InPlacePeakRate": 30.0,
        "CenterPlaceField": -5000.0,
        "SigmaPlaceField": 50000000,
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
dt = 0.1

pyr = genloss.SpatialThetaGenerators(params, mask=[True,])
pyr.precomute()


t = tf.range(0, 10000.0, dt, dtype=tf.float64)

firings = pyr.get_firings( tf.reshape(t, shape=[-1, 1]) ) * 0.001 * dt
###############################################################
pre_types = ["CA1 Pyramidal", "CA1 Basket"] #  "CA1 O-LM",
post_type = "CA1 Basket"
synparams = pd.read_csv("../parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv")
synparams.rename({"g" : "gsyn_max", "u" : "Uinc", "Connection Probability":"pconn"}, axis=1, inplace=True)

synparams['Erev'] = np.zeros( len(synparams), dtype=np.float64)
#print(synparams.keys())

neurons_params = pd.read_csv("../parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv", header=0, names=["Neuron Type", "E/I", "Izh C"], index_col=False)
neurons_params.rename({"Neuron Type" : "Presynaptic Neuron Type"}, axis=1, inplace=True)
#print(neurons_params.head(5))

selected_synparam =  synparams.loc[ (synparams["Presynaptic Neuron Type"].isin(pre_types)) & (synparams["Postsynaptic Neuron Type"] == post_type) ]

selected_synparam = selected_synparam.merge(neurons_params, how="left", on="Presynaptic Neuron Type", copy=True)


for idx, row in selected_synparam.iterrows():
    if row["E/I"].strip() == "i":
        selected_synparam.at[idx, 'Erev'] = -75.0
selected_synparam["pconn"] = 1.0

synparam = {}
for key in selected_synparam.keys():
    synparam[key] = np.asarray(selected_synparam[key])
synparam["Cm"] = 0.144 #!!!!!!!!!!! взять из параметров нейронов

## pprint(synparam)

input_shape = [1, None, 2]
synapses_layer = RNN(TsodycsMarkramSynapse(synparam, dt=0.1, mask=None), return_sequences=True, stateful=True)

model = tf.keras.Sequential()
model.add(synapses_layer)
model.build(input_shape=input_shape)

#########################################
X = np.zeros([1, len(t), 2], dtype=np.float32)
X[0, :, :] = firings.numpy() # !!!!


Y = model.predict(X)

fig, axes = plt.subplots(nrows=3)
axes[0].plot(t, firings)
axes[1].plot(t, Y[0, :, 0])
axes[2].plot(t, Y[0, :, 1])
axes[2].set_ylim(0, 80)

plt.show()