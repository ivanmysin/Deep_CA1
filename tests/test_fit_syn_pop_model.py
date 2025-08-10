import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, RNN, Input
import h5py
from synapses_layers import TsodycsMarkramSynapse
import genloss
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tensorflow.keras.saving import load_model

params = [

    {
        "R": 0.25,
        "OutPlaceFiringRate": 0.5,
        "OutPlaceThetaPhase": 3.14,
        "InPlacePeakRate": 30.0,
        "CenterPlaceField": -500000.0,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0,  # np.deg2rad(10) * 10 * 0.001,
        "PrecessionOnset": -1.57,
        "ThetaFreq": 8.0,
    },

    # {
    #     "R": 0.25,
    #     "OutPlaceFiringRate": 0.5,  # Хорошо бы сделать лог-нормальное распределение
    #     "OutPlaceThetaPhase": 3.14,  # DV
    #
    #     "InPlacePeakRate": 30,  # Хорошо бы сделать лог-нормальное распределение
    #     "CenterPlaceField": -10000,
    #     "SigmaPlaceField": 500,
    #
    #
    #     "SlopePhasePrecession": 0.0,  # DV
    #     "PrecessionOnset": 3.14,
    #
    #     "ThetaFreq": 8.0,
    # },

    {
        "R": 0.25,
        "OutPlaceFiringRate": 40,  # Хорошо бы сделать лог-нормальное распределение
        "OutPlaceThetaPhase": 3.14 ,  # DV

        "InPlacePeakRate": 30,  # Хорошо бы сделать лог-нормальное распределение
        "CenterPlaceField": -10000,
        "SigmaPlaceField": 500,

        "SlopePhasePrecession": 0.0,  # DV
        "PrecessionOnset": 3.14,

        "ThetaFreq": 8.0,
    },


]
dt = 0.5

generators = genloss.SpatialThetaGenerators(params)


###############################################################
pre_types = ["CA1 Pyramidal", "CA1 O-LM"] # "CA3 Pyramidal",  "
post_type = "CA1 Basket"
synparams = pd.read_csv("../parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv")
synparams.rename({"g" : "gsyn_max", "u" : "Uinc", "Connection Probability":"pconn"}, axis=1, inplace=True)

synparams['Erev'] = np.zeros( len(synparams), dtype=np.float64)
#print(synparams.keys())

neurons_params = pd.read_csv("../parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv", header=0, usecols=["Neuron Type", "E/I", "Izh C", "Izh Vr", "Izh k", "Izh Vt"])
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

Vrest = float( neurons_params[neurons_params["Presynaptic Neuron Type"] == post_type]["Izh Vr"].values[0] )
k = float( neurons_params[neurons_params["Presynaptic Neuron Type"] == post_type]["Izh k"].values[0] )
Vt = float( neurons_params[neurons_params["Presynaptic Neuron Type"] == post_type]["Izh Vt"].values[0] )

synparam["gl"] = k * (Vt - Vrest) * 0.001
#synparam["gsyn_max"][-2] = 3000.0
synparam["Cm"] *= 0.001
synparam["gsyn_max"][0] = 1.5
synparam["gsyn_max"][1] = 1.5
synparam["pconn"][:] = 1.0
pprint(synparam)

#input_shape = [1, None, len(params)]
conn_mask = np.ones(len(params), dtype='bool')
synapses_layer = RNN(TsodycsMarkramSynapse(synparam, dt=dt, mask=conn_mask), return_sequences=True, stateful=True)
population_model = load_model(f"../pretrained_models/{post_type}.keras", custom_objects={'square': tf.keras.ops.square})
for layer in population_model.layers:
    layer.trainable = False


t = tf.range(0, 5000.0, dt, dtype=tf.float32)
t = tf.reshape(t, shape=(1, -1, 1))
generators_firings = generators(t) #* 0.001 * dt
generators_firings = generators_firings.numpy()
# Esynt = synapses_layer(generators_firings)
# pop_firings = population_model(Esynt)
#
# target_pop_firings = pop_firings.numpy() * 0.1
#
# print(target_pop_firings.shape)



input_layer = Input(shape=(None, len(params)), batch_size=1)
synapses_layer = synapses_layer(input_layer)
syn_pop_model = Model(inputs=input_layer, outputs=population_model(synapses_layer), name=f"Population_with_synapses")
# syn_pop_model = Model(inputs=input_layer, outputs=synapses_layer, name=f"Population_with_synapses")

syn_pop_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=0.1),
    loss=tf.keras.losses.LogCosh(), # MeanSquaredError()
    metrics=[tf.keras.metrics.MeanSquaredError(), ],
)

#syn_pop_model.build(input_shape=(1, None, len(params)))

print(syn_pop_model.summary())
#print(syn_pop_model.metrics)

origin_sim_results = syn_pop_model.predict(generators_firings, batch_size=1)
target_pop_firings = origin_sim_results * 1.8


generators_firings = generators_firings.reshape(10, -1, len(params))
target_pop_firings = target_pop_firings.reshape(10, -1, 1)



# print(generators_firings.shape)
# print(target_pop_firings.shape)
#
hist = syn_pop_model.fit(
    x=generators_firings,
    y=target_pop_firings,
    epochs=5,
    batch_size=1,
    verbose=2,
)

# with tf.GradientTape() as tape:
#     y_pred = syn_pop_model(generators_firings)
#     loss_value = tf.keras.losses.logcosh(target_pop_firings, y_pred) #(y_true, y_pred)
#
#     # Проверяем значения градиента
#     gradients = tape.gradient(loss_value, syn_pop_model.trainable_variables)
#     for grad, var in zip(gradients,  syn_pop_model.trainable_variables):
#         print(var.name)
#         if tf.math.is_nan(grad).numpy().any():
#             print("Found NaN gradient")
#
#         print( grad.numpy() )



generators_firings = generators_firings.reshape(1, -1, len(params))

Esynt = syn_pop_model.predict(generators_firings)

# generators_firings = generators_firings.numpy()
# generators_firings = generators_firings[0, :, :]
t = t.numpy().ravel()
Esynt = Esynt.ravel()

#Esynt = Esynt*75 - 75
# pop_firings = pop_firings.numpy().ravel()

origin_sim_results = origin_sim_results.ravel()

generators_firings = generators_firings.reshape(-1, len(params))
target_pop_firings = target_pop_firings.ravel()
#origin_sim_results = origin_sim_results*75 - 75




fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(t, generators_firings)
axes[1].plot(t, Esynt, label="optimized", linewidth=5)
axes[1].plot(t, target_pop_firings, label="target", linewidth=1)
axes[1].plot(t, origin_sim_results, label="origin", linewidth=1)
#axes[2].plot(t, pop_firings, color='red', linewidth=3)
#axes[1].set_ylim(-75, 0)
axes[1].legend(loc='upper right')

plt.show()