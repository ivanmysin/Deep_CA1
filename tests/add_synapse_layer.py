import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, RNN, Input
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
        'Vrest' : -65.0,
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

    input_layer = Input(shape=(None, 5), batch_size=1)
    synapses_layer = synapses_layer(input_layer)
    base_model = tf.keras.models.clone_model(population_model)
    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=input_layer, outputs=base_model(synapses_layer), name="Population_with_synapses")
    model.build(input_shape=input_shape)

    print("###################################")
    print(model.summary())

    X1 = np.random.rand(5).reshape(1, 1, 5)
    X2 = np.random.rand(5).reshape(1, 1, 5)
    #!!! X = np.zeros_like(X)

    Y1 = model.predict(X1)

    Y2 = model.predict(X2)

    Y = tf.concat([Y1, Y2], axis=-1)

    print(Y.shape)