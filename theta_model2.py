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
from genloss import SpatialThetaGenerators, PhaseLockingOutputWithPhase, WeightedLMSE
import myconfig

from myutils import get_net_params



def get_base_params(path_to_base_model, sheet_name):
    base_params = get_net_params(path_to_base_model)


    populations = pd.read_excel('./parameters/neurons_parameters.xlsx', sheet_name=sheet_name)
    populations.rename( {'neurons' : 'type'}, axis=1, inplace=True)
    populations = populations[populations['is_include'] == 1]

    pop_types = populations['Model_Neurons_Names'].to_list()
    pconn_mask = base_params['pconn'] == 0
    Ngenerators = sum(populations['Simulated_Type'] == 'generator')

    simulated_types = pop_types[:-Ngenerators]

    optim_parameters_neurons = pd.DataFrame( {'Neuron Type' : simulated_types, 'I_ext': base_params['I_ext'] } )

    optim_parameters_connection =  pd.DataFrame(columns=['Presynaptic Neuron Type', 'Postsynaptic Neuron Type', 'gsyn_max', 'tau_d', 'tau_r', 'tau_f', 'Uinc', 'Erev'])

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
                'Erev': base_params['e_r'][pre_idx, post_idx],
            }



            optim_parameters_connection.loc[len(optim_parameters_connection)] = p




    return optim_parameters_connection, optim_parameters_neurons


def get_params(optim_parameters_connection, optim_parameters_neurons, sheet_name):

    SIGMA_PYR2PYR_CONNECTIONS = 50  # mkm

    neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
    neurons_params.rename(
        {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

    synapses_params = optim_parameters_connection
    synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)


    populations = pd.read_excel(myconfig.FIRINGSNEURONPARAMS, sheet_name=sheet_name)
    populations = populations[populations['Npops'] > 0]

    params = {
        'I_ext' : [],
    }
    dimpopparams = {
        'dt_dim' : myconfig.DT,
        'Delta_eta' : [],
    }

    generators_params = []

    for pop_idx, pop in populations.iterrows():

        if pop['Simulated_Type'] == 'generator':
            generators_params.append(pop.to_dict())
            continue

        hippocampome_pop_type = pop['Hippocampome_Neurons_Names']
        model_pop_type = pop['Model_Neurons_Names']



        p = neurons_params[neurons_params["Neuron Type"] == hippocampome_pop_type]

        try:
            Iext_pop = optim_parameters_neurons['I_ext'][optim_parameters_neurons['Neuron Type'] == model_pop_type][0]
            params['I_ext'].append(Iext_pop)
        except KeyError:
            params['I_ext'].append(0.0)

        try:
            dimpopparams['Delta_eta'].append(pop['Delta_eta'])
        except KeyError:
            dimpopparams['Delta_eta'].append(myconfig.DELTA_ETA)

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


    for key, val in dimpopparams.items():
        dimpopparams[key] = np.asarray(val)

    NN = len(populations)
    Nsim = NN - len(generators_params)
    gsyn_max = np.zeros(shape=(NN, Nsim), dtype=np.float32)



    params['gsyn_max'] = gsyn_max
    params["e_r"] = np.zeros_like(gsyn_max)

    params['pconn'] = np.zeros_like(gsyn_max)
    params['tau_d'] = np.zeros_like(gsyn_max) + 100  # + tau_d
    params['tau_r'] = np.zeros_like(gsyn_max) + 10  # + tau_r
    params['tau_f'] = np.zeros_like(gsyn_max) + 5  # + tau_f
    params['Uinc'] = np.zeros_like(gsyn_max) + 0.5  # + Uinc

    for pre_idx, (_, pre_pop) in enumerate(populations.iterrows()):
        for post_idx, (_, post_pop) in enumerate(populations.iterrows()):
            if post_pop['Simulated_Type'] == 'generator':
                continue

            pre_type = pre_pop['Model_Neurons_Names']
            post_type = post_pop['Model_Neurons_Names']

            syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == pre_type) & (
                synapses_params['Postsynaptic Neuron Type'] == post_type)]

            if len(syn) == 0:
                continue



            if (pre_type == 'Pyramidal (deep)' and post_type == 'Pyramidal (deep)') or (pre_type == 'Pyramidal (superficial)' and post_type == 'Pyramidal (superficial)'):
                dist_anat = np.sqrt(  (pre_pop['x_anat'] - post_pop['x_anat'])**2 + (pre_pop['y_anat'] - post_pop['y_anat'])**2 )
                params['pconn'][pre_idx, post_idx] = np.exp(  -0.5  * (dist_anat / SIGMA_PYR2PYR_CONNECTIONS)**2  )

            else:
                params['pconn'][pre_idx, post_idx] = 1




            Uinc = syn['Uinc'].values[0]
            tau_r = syn['tau_r'].values[0]
            tau_f = syn['tau_f'].values[0]
            tau_d = syn['tau_d'].values[0]

            gsyn = syn['gsyn_max'].values[0]
            Erev = syn['Erev'].values[0]

            params['Uinc'][pre_idx, post_idx] = Uinc
            params['tau_r'][pre_idx, post_idx] = tau_r
            params['tau_f'][pre_idx, post_idx] = tau_f
            params['tau_d'][pre_idx, post_idx] = tau_d

            params['gsyn_max'][pre_idx, post_idx] = gsyn
            params['e_r'][pre_idx, post_idx] = Erev

    params_dimless = izhs_lib.dimensional_to_dimensionless_all(dimpopparams)

    params = params | params_dimless


    for p in generators_params:
        p['ThetaFreq'] = myconfig.ThetaFreq

    populations['ThetaFreq'] = myconfig.ThetaFreq

    target_params = populations[ (populations['Simulated_Type'] == 'simulated')  ]


    output_masks = {
        'full_output' : np.asarray( populations['Hippocampome_Neurons_Names'] == 'CA1 Pyramidal', dtype=bool)[:Nsim],
        'phase_output' : np.asarray( populations['Hippocampome_Neurons_Names'] != 'CA1 Pyramidal', dtype=bool )[:Nsim],
    }

    return params, generators_params, target_params, output_masks
########################################################################
def get_model(params, generators_params, dt, output_masks):
    input = Input(shape=(None, 1), batch_size=1)

    generators = SpatialThetaGenerators(generators_params)(input)
    net_layer = RNN(MeanFieldNetwork(params, dt_dim=dt, use_input=True),
                    return_sequences=True, stateful=True,
                    name="firings_outputs")(generators)


    phase_locking_layer = PhaseLockingOutputWithPhase(ThetaFreq=myconfig.ThetaFreq, dt=dt)(net_layer)

    outputs = [net_layer, phase_locking_layer]
    big_model = Model(inputs=input, outputs=outputs)

    mse_full_output = WeightedLMSE(output_masks['full_output'])
    phase_locking_output = WeightedLMSE(output_masks['phase_output'])

    big_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=myconfig.LEARNING_RATE, clipvalue=10.0),
        loss = [mse_full_output, phase_locking_output],
        loss_weights = [1.0, 1.0]
    )

    return big_model

def get_dataset(target_params, dt, batch_len, nbatches, output_masks):
    duration = int(batch_len * nbatches * dt)


    target_params['SlopePhasePrecession'] = target_params['SlopePhasePrecession'] / 360 * 2 * np.pi * myconfig.V_AN / 1000

    target_params['CenterPlaceField'] = target_params['CenterPlaceField'] / myconfig.V_AN  * 1000
    target_params['SigmaPlaceField'] = target_params['SigmaPlaceField'] / myconfig.V_AN  * 1000

    target_params_full_outputs = target_params.to_dict('list')  #[output_masks['full_output']]

    generators = SpatialThetaGenerators( target_params_full_outputs )
    t = tf.reshape(tf.range(0, duration, dt, dtype=myconfig.DTYPE), shape=(1, -1, 1))

    target_firings = generators(t)


    Rs = target_params['R'].values.reshape(1, -1)
    phases = target_params['OutPlaceThetaPhase'].values.reshape(1, -1)
    MeanFirings = target_params['OutPlaceThetaPhase'].values.reshape(1, 1, -1)


    Yphase_output = np.stack( [Rs * np.cos(phases), Rs * np.sin(phases)], axis=1) * MeanFirings

    # Yphase_output = Yphase_output[:, :, output_masks['phase_output']]


    print(Yphase_output.shape)

    Y = {
        'full_output' : target_firings.numpy().reshape(nbatches, batch_len, -1),
        'phase_output' : Yphase_output,
    }

    X = t.numpy().reshape(nbatches, batch_len, 1)


    return X, Y




########################################################################
SOURCE_MODEL = myconfig.OUTPUTSPATH_MODELS + 'theta_model_5000.keras'
batch_len = 12000
nbatches = 50

optim_parameters_connection, optim_parameters_neurons = get_base_params(SOURCE_MODEL, 'local_model')
params, generators_params, target_params, output_masks = get_params(optim_parameters_connection, optim_parameters_neurons, sheet_name='full_local_model')

Xtrain, Ytrain = get_dataset(target_params, myconfig.DT, batch_len, nbatches, output_masks)




'''


with h5py.File(myconfig.OUTPUTSPATH + 'dataset.h5', mode='w') as dfile:
    dfile.create_dataset('Xtrain', data=Xtrain)
    dfile.create_dataset('Ytrain', data=Ytrain)
'''



model = get_model(params, generators_params, myconfig.DT, output_masks)



checkpoint_filepath = myconfig.OUTPUTSPATH_MODELS + 'full_local_model_{epoch:02d}.keras'
# filename_template = 'full_local_firings_{epoch:02d}.h5'

Nepoches4modelsaving = 2 * len(Xtrain) + 1


callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='loss',
            mode='auto',
            save_best_only=False,
            save_freq = 'epoch'),

        # SaveFirings( firing_model=model,
        #              t_full=Xtrain.reshape(1, -1, 1),
        #              path=myconfig.OUTPUTSPATH_FIRINGS,
        #              filename_template=filename_template,
        #              save_freq = 10),

        TerminateOnNaN(),
]

history = model.fit(x=Xtrain, y=Ytrain, epochs=10000, verbose=2, batch_size=1, callbacks=callbacks)

#Ypred = model.predict(Xtrain, batch_size=1)
with h5py.File(myconfig.OUTPUTSPATH + 'full_local_history.h5', mode='w') as dfile:
    dfile.create_dataset('loss', data=history.history['loss'])


