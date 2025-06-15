import os
os.chdir("../")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN, Input
from tensorflow.keras.models import Model



import izhs_lib
from genloss import SpatialThetaGenerators
from mean_field_class import MeanFieldNetwork
from pprint import pprint

generators_params = [
    {
        "R": 0.25,
        "OutPlaceFiringRate": 8.5,
        "OutPlaceThetaPhase": 3.14,
        "InPlacePeakRate": 8.0,
        "CenterPlaceField": -5000.0,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0, # np.deg2rad(10)*10 * 0.001,
        "PrecessionOnset":  -1.57,
        "ThetaFreq": 8.0,
    },

    {
        "R": 0.25,
        "OutPlaceFiringRate": 8.5,
        "OutPlaceThetaPhase": 3.14,
        "InPlacePeakRate": 8.0,
        "CenterPlaceField": -5000.0,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0,  # np.deg2rad(10)*10 * 0.001,
        "PrecessionOnset": np.nan,  # -1.57,
        "ThetaFreq": 5.0,
    },
]

NN = 2
Ninps = len(generators_params)
dt_dim = 0.01

genrators = SpatialThetaGenerators(generators_params)
t = tf.range(0, 2000.0, dt_dim, dtype=tf.float32)
t = tf.reshape(t, shape=(1, -1, 1))

genetators_firings = genrators(t)
#print(tf.shape(genetators_firings))


dim_izh_params = {
    "V0": -57.63,
    "U0": 0.0,

    "Cm": 114,  # * pF,
    "k": 1.19,  # * mS
    "Vrest": -57.63,  # * mV,
    "Vth": -35.53,  # *mV, # np.random.normal(loc=-35.53, scale=4.0, size=NN) * mV,  # -35.53*mV,
    "Vpeak": 21.72,  # * mV,
    "Vmin": -48.7,  # * mV,
    "a": 0.005,  # * ms ** -1,
    "b": 0.22,  # * mS,
    "d": 2,  # * pA,

    "Iext": 200,  # pA
}

# Словарь с константами
cauchy_dencity_params = {
    'Delta_eta': 15,  # 0.02,
    'bar_eta': 0.0,  # 0.191,
}

dim_izh_params = dim_izh_params | cauchy_dencity_params
izh_params = izhs_lib.dimensional_to_dimensionless(dim_izh_params)
izh_params['dts_non_dim'] = izhs_lib.transform_T(dt_dim, dim_izh_params['Cm'], dim_izh_params['k'],
                                                 dim_izh_params['Vrest'])

for key, val in izh_params.items():
    izh_params[key] = np.zeros(NN, dtype=np.float32) + val

## synaptic static variables
tau_d = 6.02  # ms
tau_r = 359.8  # ms
tau_f = 21.0  # ms
Uinc = 0.25

gsyn_max = np.zeros(shape=(NN + Ninps, NN), dtype=np.float32)
gsyn_max[3, 0] = 0
# gsyn_max[1, 0] = 15


pconn = np.zeros(shape=(NN + Ninps, NN), dtype=np.float32)
pconn[3, 0] = 1
# pconn[1, 0] = 1

Erev = np.zeros(shape=(NN + Ninps, NN), dtype=np.float32) - 75
Erev[3, 0] = 0
e_r = izhs_lib.transform_e_r(Erev, dim_izh_params['Vrest'])

izh_params['gsyn_max'] = gsyn_max
izh_params['pconn'] = pconn
izh_params['e_r'] = np.zeros_like(gsyn_max) + e_r
izh_params['tau_d'] = np.zeros_like(gsyn_max) + tau_d
izh_params['tau_r'] = np.zeros_like(gsyn_max) + tau_r
izh_params['tau_f'] = np.zeros_like(gsyn_max) + tau_f
izh_params['Uinc'] = np.zeros_like(gsyn_max) + Uinc


izh_params['nmda'] = {}

izh_params['nmda']['pconn_nmda'] = np.zeros(shape=(NN + Ninps, NN), dtype=np.float32)
izh_params['nmda']['pconn_nmda'][3, 0] = 1

izh_params['nmda']['Mgb'] = 0.27027027027027023
izh_params['nmda']['av_nmda'] = 0.062 * np.abs(dim_izh_params['Vrest'])


izh_params['nmda']['gsyn_max_nmda'] = np.zeros(shape=(NN + Ninps, NN), dtype=np.float32)
izh_params['nmda']['gsyn_max_nmda'][3, 0] = 500

izh_params['nmda']['tau1_nmda'] = 2.3
izh_params['nmda']['tau2_nmda'] = 150





meanfieldlayer = MeanFieldNetwork(izh_params, dt_dim=dt_dim, use_input=True)
meanfieldlayer_rnn = RNN(meanfieldlayer, return_sequences=True, stateful=True)

input_layer = Input(shape=(None, Ninps), batch_size=1)
output = meanfieldlayer_rnn(input_layer)

model = Model(inputs=input_layer, outputs=output)

rates = model.predict(genetators_firings)

rates = rates.reshape(-1, NN)

t = t.numpy().ravel()

print(rates.shape)
# genetators_firings
plt.plot(t, rates)
plt.show()


# plt.plot(t[0, :, 0], genetators_firings[0, :, 0])
# plt.plot(t[0, :, 0], genetators_firings[0, :, 1])
# #plt.plot(t[0, :, 0], firings[0, :, 1])
# plt.show()