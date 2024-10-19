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
        'Erev_min' : -75.0,
        'Erev_max' : 0.0,
    }
    dt = 0.1
    mask = np.ones(Ns, dtype=bool)

    input_shape = (1, None, Ns)

    synapses = TsodycsMarkramSynapse(params, dt=dt, mask=mask)

    synapses_layer = RNN(synapses, return_sequences=True, stateful=True)

    population_model = load_model("../pretrained_models/CA1 Basket.keras")


    model = tf.keras.Sequential()
    model.add(synapses_layer)

    for layer in population_model.layers:
        model.add( tf.keras.models.clone_model(layer) )


    for layer in model.layers[1:]:
        layer.trainable = False


    model.build(input_shape=input_shape)

    print(model.summary())

    X = np.random.rand(50).reshape(1, 10, 5)
    #!!! X = np.zeros_like(X)

    Y = model.predict(X)

    print(Y)