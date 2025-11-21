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

from myutils import get_net_params, get_gen_params

def get_base_params(path_to_base_model, sheet_name):
    base_params = get_net_params(SOURCE_MODEL)

    populations = pd.read_excel('./parameters/neurons_parameters.xlsx', sheet_name='local_model')
    populations.rename( {'neurons' : 'type'}, axis=1, inplace=True)
    populations = populations[populations['Npops'] > 0]

    pop_types = populations['Hippocampome_Neurons_Names'].to_list()
    pconn_mask = base_params['pconn'] == 0
    Ngenerators = sum(populations['Simulated_Type'] == 'generator')

    simulated_types = pop_types[:-Ngenerators]
    optim_parameters_neurons = pd.DataFrame( {'Neuron Type' : simulated_types, 'I_ext': base_params['I_ext'] } )

    optim_parameters_connection =  pd.DataFrame(columns=['Presynaptic Neuron Type', 'Postsynaptic Neuron Type', 'gsyn_max', 'tau_d', 'tau_r', 'tau_f', 'Uinc'])

    for pre_idx, presynaptic_type in enumerate(pop_types):
        for post_idx, postsynaptic_type in enumerate(pop_types[:-Ngenerators]):


            if pconn_mask[pre_idx, post_idx]:
                continue

            p = {
                'Presynaptic Neuron Type': presynaptic_type,
                'Postsynaptic Neuron Type': postsynaptic_type,
                'gsyn_max' : base_params['gsyn_max'][pre_idx, post_idx],
                'tau_d' : base_params['tau_d'][pre_idx, post_idx],
                'tau_r': base_params['tau_r'][pre_idx, post_idx],
                'tau_f': base_params['tau_f'][pre_idx, post_idx],
                'Uinc': base_params['Uinc'][pre_idx, post_idx],
            }

            optim_parameters_connection.loc[len(optim_parameters_connection)] = p


    return optim_parameters_connection, optim_parameters_neurons


def get_params():




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
        loss = tf.keras.losses.MeanSquaredLogarithmicError(), # LogCosh(),    #
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
SOURCE_MODEL = myconfig.OUTPUTSPATH_MODELS + 'theta_model_5000.keras'
batch_len = 12000
nbatches = 20

optim_parameters_connection, optim_parameters_neurons = get_base_params(SOURCE_MODEL, 'local_model')

print(optim_parameters_neurons)

params, generators_params, target_params = get_params(optim_parameters_connection)

'''
Xtrain, Ytrain = get_dataset(target_params, myconfig.DT, batch_len, nbatches)

with h5py.File(myconfig.OUTPUTSPATH + 'dataset.h5', mode='w') as dfile:
    dfile.create_dataset('Xtrain', data=Xtrain)
    dfile.create_dataset('Ytrain', data=Ytrain)


model = get_model(params, generators_params, myconfig.DT)

checkpoint_filepath = myconfig.OUTPUTSPATH_MODELS + 'base_big_model_{epoch:02d}.keras'
filename_template = 'base_firings_{epoch:02d}.h5'

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
with h5py.File(myconfig.OUTPUTSPATH + 'history.h5', mode='w') as dfile:
    dfile.create_dataset('loss', data=history.history['loss'])

'''
