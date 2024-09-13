import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN
from tensorflow.keras.saving import load_model
import h5py
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from synapses_layers import TsodycsMarkramSynapse

if True:
    Ns = 5
    params = {
        "gsyn_max" : np.zeros(Ns, dtype=np.float32) + 1.5,
        "Uinc" :  np.zeros(Ns, dtype=np.float32) + 0.5,
        "tau_r" : np.zeros(Ns, dtype=np.float32) + 1.5,
        "tau_f" : np.zeros(Ns, dtype=np.float32) + 1.5,
        "tau_d" : np.zeros(Ns, dtype=np.float32) + 1.5,
        'pconn' : np.zeros(Ns, dtype=np.float32) + 1.0,
        'Erev' : np.zeros(Ns, dtype=np.float32),
        'Cm' : 0.114,
    }
    dt = 0.1
    mask = np.ones(Ns, dtype=bool)

    input_shape = (1, None, Ns)

    synapses = TsodycsMarkramSynapse(params, dt=dt, mask=mask)

    synapses_layer = RNN(synapses, return_sequences=True, stateful=True)

    population_model = load_model("../pretrained_models/pv_bas.keras")


    model = tf.keras.Sequential()
    model.add(synapses_layer)

    model.add( tf.keras.models.clone_model(population_model.layers[0]) )
    model.add( tf.keras.models.clone_model(population_model.layers[1]) )

    model.layers[1].trainable = False
    model.layers[2].trainable = False

    model.build(input_shape=input_shape)

    print(model.summary())

    X = np.random.rand(150).reshape(1, 10, 5)

    Y = model.predict(X)

    print(Y.shape)