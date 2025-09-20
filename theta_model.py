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
from genloss import SpatialThetaGenerators, PhaseLockingOutput,  WeightedMSE, WeightedLMSE, FiringsMeanOutRanger
# from genloss import SpatialThetaGenerators, CommonOutProcessing, PhaseLockingOutputWithPhase, PhaseLockingOutput, RobastMeanOut, FiringsMeanOutRanger, Decorrelator

import myconfig


def get_params():

    # TestPopulation = 'CA1 O-LM'
    # output_masks = {
    #     'full_target' : [],
    #     'only_R' : [],
    # }


    neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
    neurons_params.rename(
        {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

    synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
    synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)

    potconn_file_path = './parameters/PetentialConnections.csv'
    potential_connections = pd.read_csv(potconn_file_path)
    potential_connections['Presynaptic Neuron Type'] = potential_connections['Presynaptic Neuron Type'].str.strip()
    potential_connections['Postsynaptic Neuron Type'] = potential_connections['Postsynaptic Neuron Type'].str.strip()

    populations = pd.read_excel(myconfig.FIRINGSNEURONPARAMS, sheet_name='verified_theta_model')
    # populations.rename( {'neurons' : 'type'}, axis=1, inplace=True)
    populations = populations[populations['Npops'] > 0]

    params = {}
    dimpopparams = {
        'dt_dim' : myconfig.DT,
        'Delta_eta' : myconfig.DELTA_ETA,
        'I_ext' : [],
    }

    generators_params = []

    for pop_idx, pop in populations.iterrows():

        if pop['Simulated_Type'] == 'generator':
            generators_params.append(pop.to_dict())
            continue

        hippocampome_pop_type = pop['Hippocampome_Neurons_Names']

        for m in output_masks.values():
            m.append(True)

        if hippocampome_pop_type == TestPopulation:
            # output_masks['only_R'][-1] = True
            output_masks['full_target'][-1] = False
        # else:
        #     output_masks['only_R'][-1] = False
        #     output_masks['full_target'][-1] = True



        p = neurons_params[neurons_params["Neuron Type"] == hippocampome_pop_type]

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

    for pre_idx, (_, pre_pop) in enumerate(populations.iterrows()):
        for post_idx, (_, post_pop) in enumerate(populations.iterrows()):
            if post_pop['Simulated_Type'] == 'generator':
                continue

            pre_type = pre_pop['Hippocampome_Neurons_Names']
            post_type = post_pop['Hippocampome_Neurons_Names']

            syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == pre_type) & (
                    synapses_params['Postsynaptic Neuron Type'] == post_type)]

            if len(syn) == 0:
                # print("Connection from ", pre_type, "to", post_type, "not finded!")

                syn = potential_connections[(potential_connections['Presynaptic Neuron Type'] == pre_type) & (
                        potential_connections['Postsynaptic Neuron Type'] == post_type)]

                if len(syn) == 0:
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


    for p in generators_params:
        p['ThetaFreq'] = myconfig.ThetaFreq

    populations['ThetaFreq'] = myconfig.ThetaFreq

    target_params = populations[ (populations['Simulated_Type'] == 'simulated')  ]


    # target_params.loc[ target_params['Hippocampome_Neurons_Names'] == TestPopulation, "R"] = 0

    return params, generators_params, target_params, output_masks
########################################################################
def get_model(params, generators_params, dt, target_params):
    input = Input(shape=(None, 1), batch_size=1)

    mean_firings_rates = [fr for fr in target_params['OutPlaceFiringRate'] ]

    generators = SpatialThetaGenerators(generators_params)(input)
    net_layer = RNN(MeanFieldNetwork(params, dt_dim=dt, use_input=True),
                    return_sequences=True, stateful=True,
                    # activity_regularizer=FiringsMeanOutRanger(HighFiringRateBound=200.0),
                    name="firings_outputs")(generators)

    only_modulation_output = PhaseLockingOutput(
                                    mean_firings_rates,
                                    ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT,
                                    name='only_modulation_output')(net_layer)

    # outputs = generators # net_layer  #
    outputs = [net_layer, only_modulation_output]  # generators #
    big_model = Model(inputs=input, outputs=outputs)

    firing_model = Model(inputs=input, outputs=net_layer)

    lmse_loss = tf.keras.losses.MeanSquaredLogarithmicError()   # # WeightedLMSE(output_masks['full_target'])
    # mse_loss = tf.keras.losses.MeanSquaredError() # WeightedMSE(output_masks['only_R'])

    big_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=myconfig.LEARNING_RATE, clipvalue=10.0),
        loss = [lmse_loss, lmse_loss],
        loss_weights = [1.0, 0.1],
    )

    return big_model, firing_model

def get_dataset(target_params, dt, batch_len, nbatches):
    duration = int(batch_len * nbatches * dt)

    generators = SpatialThetaGenerators(target_params)
    t = tf.reshape(tf.range(0, duration, dt, dtype=myconfig.DTYPE), shape=(1, -1, 1))

    target_firings = generators(t)

    #print(target_firings[0, :10, ])

    X = t.numpy().reshape(nbatches, batch_len, 1)
    Y = target_firings.numpy().reshape(nbatches, batch_len, -1)



    Rs =  target_params['R'].values.astype(myconfig.DTYPE).reshape(1, 1, Y.shape[-1])
    Ytrain_R = np.zeros(shape=(nbatches, 1, Y.shape[-1]), dtype=myconfig.DTYPE) + Rs

    Y = [Y, Ytrain_R]

    return X, Y




########################################################################
IS_CREATE_MODEL = True
checkpoint_filepath = myconfig.OUTPUTSPATH_MODELS + 'theta_model_{epoch:02d}.keras'  # 'add_R_theta_model_{epoch:02d}.keras' # 'verified_theta_model_{epoch:02d}.keras'
filename_template =  'theta_firings_{epoch:02d}.h5'  # 'add_R_theta_firings_{epoch:02d}.h5'   #'verified_theta_firings_{epoch:02d}.h5'

model_path = myconfig.OUTPUTSPATH_MODELS + 'add_R_theta_model_10000.keras'
initial_epoch = 10000
Epoches = 10000

if IS_CREATE_MODEL:
    batch_len = 20000 #!!! 12500
    nbatches = 12 #!!! 20
    params, generators_params, target_params, output_masks = get_params()



    Xtrain, Ytrain = get_dataset(target_params, myconfig.DT, batch_len, nbatches)

    with h5py.File(myconfig.OUTPUTSPATH + 'dataset.h5', mode='w') as dfile:
        dfile.create_dataset('Xtrain', data=Xtrain)
        dfile.create_dataset('Ytrain', data=Ytrain[0])
        dfile.create_dataset('Ytrain_R', data=Ytrain[1])


    model, firing_model = get_model(params, generators_params, myconfig.DT, target_params)

    initial_epoch = 0

else:

    model = load_model(model_path)

    firings_outputs_layer = model.get_layer('firings_outputs').output
    firing_model = Model(inputs=model.input, outputs=firings_outputs_layer)

    with h5py.File(myconfig.OUTPUTSPATH + 'dataset.h5', mode='r') as dfile:
        Xtrain = dfile['Xtrain'][:]
        Ytrain_1 = dfile['Ytrain'][:]
        Ytrain_2 = dfile['Ytrain_R'][:]

    Ytrain = [Ytrain_1, Ytrain_2]

    Epoches = Epoches + initial_epoch




Nepoches4modelsaving = 2 * len(Xtrain) + 1


callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='loss',
            mode='auto',
            save_best_only=False,
            save_freq = 'epoch'),

        SaveFirings( firing_model=firing_model,
                     t_full=Xtrain.reshape(1, -1, 1),
                     path=myconfig.OUTPUTSPATH_FIRINGS,
                     filename_template=filename_template,
                     save_freq = 10),

        TerminateOnNaN(),
]

history = model.fit(x=Xtrain, y=Ytrain, epochs=Epoches, verbose=2, batch_size=1, callbacks=callbacks, initial_epoch=initial_epoch)

#Ypred = model.predict(Xtrain, batch_size=1)
with h5py.File(myconfig.OUTPUTSPATH + 'theta_history.h5', mode='w') as dfile:
    dfile.create_dataset('loss', data=history.history['loss'])


