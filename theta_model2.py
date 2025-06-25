import numpy as np
import tensorflow as tf
import pandas as pd
import h5py
import izhs_lib
from pprint import pprint

import os


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RNN, Layer
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN

from mean_field_class import MeanFieldNetwork, SaveFirings
from genloss import SpatialThetaGenerators
import myconfig


def get_params():

    neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
    neurons_params.rename(
        {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

    change_columns = {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b',
                      'Izh d': 'd',
                      'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}.values()

    copyed_indx = neurons_params.index[neurons_params['Neuron Type'] == 'CA1 Basket']
    tril_indx = neurons_params.index[neurons_params['Neuron Type'] == 'CA1 Trilaminar']

    for col in change_columns:
        neurons_params.loc[tril_indx, col] = neurons_params.loc[copyed_indx, col]


    synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
    synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)

    populations = pd.read_excel(myconfig.FIRINGSNEURONPARAMS, sheet_name='theta_model')
    populations.rename( {'neurons' : 'type'}, axis=1, inplace=True)
    populations = populations[populations['Npops'] > 0]

    params = {}
    dimpopparams = {
        'dt_dim' : myconfig.DT,
        'Delta_eta' : myconfig.DELTA_ETA,
        'I_ext' : [],
    }

    generators_params = []

    for pop_idx, pop in populations.iterrows():

        pop_type = pop['type']
        if 'generator' in pop_type:
            generators_params.append(pop.to_dict())
            continue

        p = neurons_params[neurons_params["Neuron Type"] == pop_type]

        try:
            dimpopparams['I_ext'].append(pop['I_ext'])
        except KeyError:
            dimpopparams['I_ext'].append(0.0)

        for key in p:
            val = p[key].values[0]
            try:
                val = float(val)
            except ValueError:
                continue

            if key in dimpopparams.keys():
                dimpopparams[key].append(val)
            else:
                dimpopparams[key] = [val, ]


    #pprint(generators_params)
    for key, val in dimpopparams.items():
        dimpopparams[key] = np.asarray(val)

    #pprint(dimpopparams)

    NN = len(populations)
    Nsim = NN - len(generators_params)
    gsyn_max = np.zeros(shape=(NN, Nsim), dtype=np.float32)

    #print(gsyn_max.shape)

    dimpopparams['gsyn_max'] = gsyn_max
    dimpopparams["Erev"] = np.zeros_like(gsyn_max) - 75.0

    params['pconn'] = np.zeros_like(gsyn_max)
    params['tau_d'] = np.zeros_like(gsyn_max) + 100  # + tau_d
    params['tau_r'] = np.zeros_like(gsyn_max) + 10  # + tau_r
    params['tau_f'] = np.zeros_like(gsyn_max) + 5  # + tau_f
    params['Uinc'] = np.zeros_like(gsyn_max) + 0.5  # + Uinc

    #params['nmda'] = {}

    for pre_idx, (_, pre_pop) in enumerate(populations.iterrows()):
        for post_idx, (_, post_pop) in enumerate(populations.iterrows()):
            if 'generator' in post_pop['type']:
                continue

            pre_type = pre_pop['type']
            post_type = post_pop['type']

            if pre_type == "CA3_generator":
                pre_type = 'CA3 Pyramidal'

            if pre_type == "CA1 Pyramidal_generator":
                pre_type = 'CA1 Pyramidal'

            if pre_type == "MEC_generator":
                pre_type = 'EC LIII Pyramidal'

            if pre_type == "LEC_generator":
                pre_type = 'EC LIII Pyramidal'



            syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == pre_type) & (
                    synapses_params['Postsynaptic Neuron Type'] == post_type)]

            if len(syn) == 0:
                print("Connection from ", pre_type, "to", post_type, "not finded!")
                continue

            params['pconn'][pre_idx, post_idx] = 1 # syn['pconn'].values[0]
            Uinc = syn['Uinc'].values[0]
            tau_r = syn['tau_r'].values[0]
            tau_f = syn['tau_f'].values[0]
            tau_d = syn['tau_d'].values[0]

            if neurons_params[neurons_params['Neuron Type'] == pre_type]['E/I'].values[0] == "e":
                Erev = 0
            elif neurons_params[neurons_params['Neuron Type'] == pre_type]['E/I'].values[0] == "i":
                Erev = -75.0

            gsyn_max = syn['gsyn_max'].values[0]

            params['Uinc'][pre_idx, post_idx] = Uinc
            params['tau_r'][pre_idx, post_idx] = tau_r
            params['tau_f'][pre_idx, post_idx] = tau_f
            params['tau_d'][pre_idx, post_idx] = tau_d
            dimpopparams['Erev'][pre_idx, post_idx] = Erev
            dimpopparams['gsyn_max'][pre_idx, post_idx] = gsyn_max

    params_dimless = izhs_lib.dimensional_to_dimensionless_all(dimpopparams)

    params = params | params_dimless

    # params['nmda']['pconn_nmda'] = np.sign(params['e_r'], dtype=np.float32) #!!!
    # params['nmda']['Mgb'] = 0.27027027027027023
    #
    # params['nmda']['av_nmda'] = 0.062 * np.abs(dimpopparams['Vrest']).reshape(1, -1)
    #
    # params['nmda']['gsyn_max_nmda'] = np.zeros_like(gsyn_max) + 15000.0
    #
    # params['nmda']['tau1_nmda'] = np.zeros_like(gsyn_max) + 2.3
    # params['nmda']['tau2_nmda'] = np.zeros_like(gsyn_max) + 150.0


    for p in generators_params:
        p['ThetaFreq'] = myconfig.ThetaFreq

    populations['ThetaFreq'] = myconfig.ThetaFreq

    target_params = populations[~populations['type'].astype(str).str.contains('generator')]

    return params, generators_params, target_params
########################################################################
def get_model(params, generators_params, dt):
    input = Input(shape=(None, 1), batch_size=1)

    generators = SpatialThetaGenerators(generators_params)(input)
    net_layer = RNN(MeanFieldNetwork(params, dt_dim=dt, use_input=True),
                    return_sequences=True, stateful=True,
                    name="firings_outputs")(generators)


    outputs = net_layer  # generators #
    big_model = Model(inputs=input, outputs=outputs)

    big_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=myconfig.LEARNING_RATE, clipvalue=10.0),
        loss = tf.keras.losses.MeanSquaredLogarithmicError(),
    )

    return big_model

def get_dataset(target_params, dt, batch_len, nbatches):
    duration = int(batch_len * nbatches * dt)

    generators = SpatialThetaGenerators(target_params)
    t = tf.reshape(tf.range(0, duration, dt, dtype=myconfig.DTYPE), shape=(1, -1, 1))

    target_firings = generators(t)

    #print(target_firings[0, :10, ])

    X = t.numpy().reshape(nbatches, batch_len, 1)
    Y = target_firings.numpy().reshape(nbatches, batch_len, -1)

    return X, Y




########################################################################
batch_len = 12000
nbatches = 20
params, generators_params, target_params = get_params()

Xtrain, Ytrain = get_dataset(target_params, myconfig.DT, batch_len, nbatches)

with h5py.File(myconfig.OUTPUTSPATH + 'dataset.h5', mode='w') as dfile:
    dfile.create_dataset('Xtrain', data=Xtrain)
    dfile.create_dataset('Ytrain', data=Ytrain)


model = get_model(params, generators_params, myconfig.DT)

checkpoint_filepath = myconfig.OUTPUTSPATH_MODELS + '2_big_model_{epoch:02d}.keras'
filename_template = '2_firings_{epoch:02d}.h5'

Nepoches4modelsaving = 2 * len(Xtrain) + 1


callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='loss',
            mode='auto',
            save_best_only=False,
            save_freq = 'epoch'),

        SaveFirings( firing_model=model,
                     t_full=Xtrain.reshape(1, -1, 1),
                     path=myconfig.OUTPUTSPATH_FIRINGS,
                     filename_template=filename_template,
                     save_freq = 10),

        TerminateOnNaN(),
]

history = model.fit(x=Xtrain, y=Ytrain, epochs=2000, verbose=2, batch_size=1, callbacks=callbacks)

#Ypred = model.predict(Xtrain, batch_size=1)
with h5py.File(myconfig.OUTPUTSPATH + '2_history.h5', mode='w') as dfile:
    dfile.create_dataset('loss', data=history.history['loss'])


