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
from genloss import SpatialThetaGenerators, MultiLoss,  PhaseLockingOutputWithPhase, WeightedLMSE, WeightedMSE, WeightedLogCoshError
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
        'Delta_eta_min' : [],
        'Delta_eta_max' : [],

    }

    generators_params = []

    for pop_idx, pop in populations.iterrows():

        if pop['Simulated_Type'] == 'generator':

            gen = pop.to_dict()

            gen['SlopePhasePrecession'] = gen['SlopePhasePrecession'] / 360 * 2 * np.pi * myconfig.V_AN / 1000

            gen['CenterPlaceField'] = gen['CenterPlaceField'] / myconfig.V_AN  * 1000
            gen['SigmaPlaceField'] = gen['SigmaPlaceField'] / myconfig.V_AN  * 1000

            generators_params.append(gen)
            continue

        hippocampome_pop_type = pop['Hippocampome_Neurons_Names']
        model_pop_type = pop['Model_Neurons_Names']



        p = neurons_params[neurons_params["Neuron Type"] == hippocampome_pop_type]

        try:
            Iext_pop = optim_parameters_neurons['I_ext'][optim_parameters_neurons['Neuron Type'] == model_pop_type].values[0]
            params['I_ext'].append(Iext_pop)
        except IndexError or KeyError:
            params['I_ext'].append(0.0)

        try:
            dimpopparams['Delta_eta'].append(pop['Delta_eta'])
            dimpopparams['Delta_eta_min'].append(pop['Delta_eta_min'])
            dimpopparams['Delta_eta_max'].append(pop['Delta_eta_max'])

        except KeyError:
            dimpopparams['Delta_eta'].append(myconfig.DELTA_ETA)
            dimpopparams['Delta_eta_min'].append(1.0)
            dimpopparams['Delta_eta_max'].append(20.0)

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

    params['nmda'] = {
        'pconn_nmda' : np.zeros_like(gsyn_max),
        'Mgb' : 0.27027027027027023,
        'av_nmda' : np.zeros_like(gsyn_max),
        'gsyn_max_nmda' : np.zeros_like(gsyn_max),
        'tau1_nmda' : 2.0,
        'tau2_nmda' : 89.0,

    }

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



            if ('Pyramidal' in pre_type) and ('Pyramidal' in post_type):
                dist_anat = np.sqrt(  (pre_pop['x_anat'] - post_pop['x_anat'])**2 + (pre_pop['y_anat'] - post_pop['y_anat'])**2 )
                params['pconn'][pre_idx, post_idx] = np.exp(  -0.5  * (dist_anat / SIGMA_PYR2PYR_CONNECTIONS)**2  )

            else:
                params['pconn'][pre_idx, post_idx] = 1




            Uinc = syn['Uinc'].values[0]
            tau_r = syn['tau_r'].values[0]
            tau_f = syn['tau_f'].values[0]
            tau_d = syn['tau_d'].values[0]




            params['Uinc'][pre_idx, post_idx] = Uinc
            params['tau_r'][pre_idx, post_idx] = tau_r
            params['tau_f'][pre_idx, post_idx] = tau_f
            params['tau_d'][pre_idx, post_idx] = tau_d


            Erev = syn['Erev'].values[0]

            if Erev > 0:
                gsyn = 0.01 * syn['gsyn_max'].values[0]

                params['nmda']['pconn_nmda'][pre_idx, post_idx] = 1.0
                params['nmda']['gsyn_max_nmda'][pre_idx, post_idx] = np.random.uniform(0.05, 0.5)

                post_name = post_pop['Hippocampome_Neurons_Names']
                vpost_rest = neurons_params[neurons_params['Neuron Type'] == post_name]['Vrest'].values[0]

                params['nmda']['av_nmda'][pre_idx, post_idx] = 0.062 * np.abs(vpost_rest)



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


    # phase_locking_layer = PhaseLockingOutputWithPhase(ThetaFreq=myconfig.ThetaFreq, dt=dt)(net_layer)
    #
    # outputs = {
    #     "full_output" : net_layer,
    #     "phase_output": phase_locking_layer,
    # }
    big_model = Model(inputs=input, outputs=net_layer)

    # mse_full_output = tf.keras.losses.MeanSquaredLogarithmicError()
    # mse_full_output = WeightedLMSE(output_masks['full_output'])
    # mse_full_output = WeightedLogCoshError(output_masks['full_output'])
    # phase_locking_output = WeightedMSE(output_masks['phase_output'])
    # phase_locking_output = WeightedLogCoshError(output_masks['phase_output'])

    # loss_funcs = {
    #     "full_output" : mse_full_output,
    #     "phase_output": phase_locking_output,
    # }

    big_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=myconfig.LEARNING_RATE, clipvalue=10.0),
        loss = tf.keras.losses.MeanSquaredLogarithmicError() # loss_funcs, # MultiLoss(), #
        # loss_weights = {
        #     "full_output" : 1.0,
        #     "phase_output": 0.25,
        # }
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


    Yphase_output = np.tile(Yphase_output, (nbatches, 1, 1))

    # Yphase_output = Yphase_output[:, :, output_masks['phase_output']]

    Y = {
        'full_output' : target_firings.numpy().reshape(nbatches, batch_len, -1),
        'phase_output' : Yphase_output,
    }

    X = t.numpy().reshape(nbatches, batch_len, 1)


    return X, Y




########################################################################
SOURCE_MODEL = myconfig.OUTPUTSPATH_MODELS + 'theta_model_5000.keras'

duration = 6000
myconfig.DT = 0.025
batch_len = int(120 / myconfig.DT)
nbatches = int(duration / batch_len / myconfig.DT)



optim_parameters_connection, optim_parameters_neurons = get_base_params(SOURCE_MODEL, 'local_model')
params, generators_params, target_params, output_masks = get_params(optim_parameters_connection, optim_parameters_neurons, sheet_name='full_local_model')

Xtrain, Ytrain = get_dataset(target_params, myconfig.DT, batch_len, nbatches, output_masks)



with h5py.File(myconfig.OUTPUTSPATH + 'dataset.h5', mode='w') as dfile:
    dfile.create_dataset('Xtrain', data=Xtrain)
    dfile.create_dataset('Y_full_outputs', data=Ytrain['full_output'])
    dfile.create_dataset('Y_phase_output', data=Ytrain['phase_output'])


model = get_model(params, generators_params, myconfig.DT, output_masks)

model.save(myconfig.OUTPUTSPATH_MODELS + 'full_local_model.keras')



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



# До обучения
# print("До обучения:", model.layers[2].cell.Delta_eta.numpy())


history = model.fit(x=Xtrain, y=Ytrain['full_output'], epochs=5000, verbose=2, batch_size=1, callbacks=callbacks)

# После обучения
# print("После обучения:", model.layers[2].cell.Delta_eta.numpy())

# Ypred = model.predict(Xtrain, batch_size=1)
# print(Ypred['phase_output'])

# L = np.log(Ypred['phase_output'] + 1.0) - np.log(Ytrain['phase_output'] + 1.0)
#
# print(L)
#
with h5py.File(myconfig.OUTPUTSPATH + 'full_local_history.h5', mode='w') as dfile:
    dfile.create_dataset('loss', data=history.history['loss'])

