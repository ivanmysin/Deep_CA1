import os

import numpy as np
from keras.src.ops import dtype

os.chdir("../")

import tensorflow as tf
import genloss
from synapses_layers import TsodycsMarkramSynapse

# mask = np.random.uniform(0, 1, 10) > 0.5
# input = 1000 * np.random.uniform(0, 10, 100).reshape(1, 10, 10)


# regularizer = genloss.RobastMeanOutRanger()
#
# layer = genloss.FrequencyFilter(mask)
# layer.activity_regularizer = regularizer
# out = layer(input)
# print(out.shape)
# print( layer.losses )


params = {
    "gsyn_max": None,
    "tau_f": None,
    "tau_d": None,
    "tau_r": None,
    "Uinc": None,
    "pconn": None,
    "Erev": None,
    "Erev_min": None,
    "Erev_max": None,
    "Cm": None,
    "dt" : None,
    "mask" : None,
}

for key in params.keys():
    params[key] = np.random.uniform(0.5, 10, 5)

params["mask"] = np.ones(5, dtype='bool')
params["gsyn_max"] = np.ones(5, dtype=np.float32)

input = np.random.uniform(0, 10, 25).reshape(1, 5, 5)

synapse = TsodycsMarkramSynapse(params)

states = synapse.get_initial_state(batch_size=1)
out = synapse(input, states)
print(synapse.losses)
synapse.add_regularization_penalties()
print(synapse.losses)

print( 0.001 * np.sum(params["gsyn_max"]**2))

