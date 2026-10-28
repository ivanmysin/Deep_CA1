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

from mean_field_class import MeanFieldNetwork, SaveFirings, MinMaxWeights
from genloss import SpatialThetaGenerators, MultiLoss,  PhaseLockingOutputWithPhase, WeightedLMSE, WeightedMSE, WeightedLogCoshError, CommonGenerator
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


def get_params(sheet_name, optim_parameters_connection=None, optim_parameters_neurons=None):

    SIGMA_PYR2PYR_CONNECTIONS = 50  # mkm

    neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
    neurons_params.rename(
        {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

    if optim_parameters_connection is None:
        synapses_params =  pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
    else:
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
            # True generators (theta/place field generators) - unchanged
            gen = pop.to_dict()
            gen['SlopePhasePrecession'] = gen['SlopePhasePrecession'] / 360 * 2 * np.pi * myconfig.V_AN / 1000
            gen['CenterPlaceField'] = gen['CenterPlaceField'] / myconfig.V_AN  * 1000
            gen['SigmaPlaceField'] = gen['SigmaPlaceField'] / myconfig.V_AN  * 1000
            generators_params.append(gen)
            continue

        # Only pyramidal cells go into simulated populations
        hippocampome_pop_type = pop['Hippocampome_Neurons_Names']
        model_pop_type = pop['Model_Neurons_Names']

        p = neurons_params[neurons_params["Neuron Type"] == hippocampome_pop_type]

        try:
            Iext_pop = optim_parameters_neurons['I_ext'][optim_parameters_neurons['Neuron Type'] == model_pop_type].values[0]
            params['I_ext'].append(Iext_pop)
        except (IndexError, KeyError, TypeError):
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

    # --- NEW: Calculate matrix sizes ---
    N_generators = len(generators_params)  # True generators + interneurons (as inputs)
    NN = len(populations)
    Nsim = NN - len(generators_params)

    gsyn_max = np.zeros(shape=(NN, Nsim), dtype=np.float32)

    params['gsyn_max'] = gsyn_max
    params["e_r"] = np.zeros_like(gsyn_max)

    params['pconn'] = np.zeros_like(gsyn_max)
    params['tau_d'] = np.zeros_like(gsyn_max) + 100
    params['tau_r'] = np.zeros_like(gsyn_max) + 10
    params['tau_f'] = np.zeros_like(gsyn_max) + 5
    params['Uinc'] = np.zeros_like(gsyn_max) + 0.5


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


                try:

                    params['nmda']['pconn_nmda'][pre_idx, post_idx] = 1.0
                    params['nmda']['gsyn_max_nmda'][pre_idx, post_idx] = np.random.uniform(0.05, 0.5)

                    post_name = post_pop['Hippocampome_Neurons_Names']
                    vpost_rest = neurons_params[neurons_params['Neuron Type'] == post_name]['Vrest'].values[0]

                    params['nmda']['av_nmda'][pre_idx, post_idx] = 0.062 * np.abs(vpost_rest)

                except KeyError:
                    pass

            params['gsyn_max'][pre_idx, post_idx] = gsyn
            params['e_r'][pre_idx, post_idx] = Erev

    params_dimless = izhs_lib.dimensional_to_dimensionless_all(dimpopparams)

    params = params | params_dimless


    for p in generators_params:
        p['ThetaFreq'] = myconfig.ThetaFreq

    populations['ThetaFreq'] = myconfig.ThetaFreq

    target_params = populations[ (populations['Simulated_Type'] == 'simulated')  ]
    output_masks = {}

    return params, generators_params, target_params, output_masks
########################################################################
def get_model(params, generators_params, dt, output_masks, path_to_model=None):
    """
    Build or load the CA1 model.
    
    If path_to_model is None: build model from scratch with given params.
    If path_to_model is provided: load model from file (for fine-tuning).
    
    Args:
        params: Network parameters (synaptic weights, neuron params, etc.)
        generators_params: Parameters for SpatialThetaGenerators (generators + interneurons)
        dt: Time step
        output_masks: Masks for output selection
        path_to_model: Path to existing .keras model file, or None to build from scratch
    
    Returns:
        Compiled Keras model
    """
    if path_to_model is not None:
        # --- Load existing model for fine-tuning ---
        print(f"Loading model from: {path_to_model}")
        model = load_model(path_to_model, custom_objects={
            'MeanFieldNetwork': MeanFieldNetwork,
            'SpatialThetaGenerators': SpatialThetaGenerators,
            'CommonGenerator': CommonGenerator,
            'MinMaxWeights': MinMaxWeights,
        })
        print(f"Model loaded successfully. Learning rate: {myconfig.LEARNING_RATE}")
        return model
    
    # --- Build model from scratch ---
    print("Building model from scratch...")
    
    # Input: time variable only
    # All firing rates (generators + interneurons) are computed by SpatialThetaGenerators
    # and fed into MeanFieldNetwork as external input via use_input=True
    input = Input(shape=(None, 1), batch_size=myconfig.BATCH_SIZE, name="time_input" )

    # Generate all input firing rates: true generators + interneurons (theta-modulated)
    generators = SpatialThetaGenerators(generators_params)(input)

    # MeanFieldNetwork receives generators as input and simulates only pyramidal cells
    # use_input=True appends generator firing rates to pyramidal self-connection firing rates
    net_layer = RNN(MeanFieldNetwork(params, dt_dim=dt, use_input=True),
                    return_sequences=True, stateful=True,
                    name="firings_outputs")(generators)

    # Output: firing rates of pyramidal cells only
    big_model = Model(inputs=input, outputs=net_layer)

    big_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=myconfig.LEARNING_RATE, clipvalue=10.0),
        loss = tf.keras.losses.MeanSquaredLogarithmicError()
    )
    
    print("Model built successfully.")
    print(f"  Pyramidal populations: {params['gsyn_max'].shape[1]}")
    print(f"  Generator populations (incl. interneurons): {generators_params.__len__()}")
    print(f"  Synaptic matrix shape: {params['gsyn_max'].shape}")
    
    return big_model

def get_dataset(target_params, dt, batch_len, nbatches, output_masks):
    # Generate training dataset
    # target_params contains only pyramidal cells (interneurons are in generators_params)
    duration = int(batch_len * nbatches * dt)


    target_params['SlopePhasePrecession'] = target_params['SlopePhasePrecession'] / 360 * 2 * np.pi * myconfig.V_AN / 1000

    target_params['CenterPlaceField'] = target_params['CenterPlaceField'] / myconfig.V_AN  * 1000
    target_params['SigmaPlaceField'] = target_params['SigmaPlaceField'] / myconfig.V_AN  * 1000

    # Generate target firing rates for pyramidal cells only
    target_params_full_outputs = target_params.to_dict('list')

    generators = SpatialThetaGenerators( target_params_full_outputs )
    t = tf.reshape(tf.range(0, duration, dt, dtype=myconfig.DTYPE), shape=(1, -1, 1))

    target_firings = generators(t)


    # --- NEW: No phase_output for interneurons (they are inputs now) ---
    # Only keep pyramidal cell targets
    Rs = target_params['R'].values.reshape(1, -1)
    phases = target_params['OutPlaceThetaPhase'].values.reshape(1, -1)
    MeanFirings = target_params['OutPlaceThetaPhase'].values.reshape(1, 1, -1)


    # Yphase_output = np.stack( [Rs * np.cos(phases), Rs * np.sin(phases)], axis=1) * MeanFirings
    #
    #
    # Yphase_output = np.tile(Yphase_output, (nbatches, 1, 1))

    # Y = {
    #     'full_output' : target_firings.numpy().reshape(nbatches, batch_len, -1),
    #     'phase_output' : Yphase_output,
    # }

    Y = target_firings.numpy().reshape(nbatches, batch_len, -1)
    X = t.numpy().reshape(nbatches, batch_len, 1)


    return X, Y




########################################################################
# --- Configuration ---
# Set path_to_model=None to build from scratch
# Set path_to_model='path/to/model.keras' to load and fine-tune
PATH_TO_MODEL = None  # Change to model path for fine-tuning, or keep None for fresh training

# Output model name (used when building from scratch)
OUTPUT_MODEL_NAME = 'pyr_only_model.keras'

# Training parameters
duration = 6000
myconfig.DT = 0.05
batch_len = int(120 / myconfig.DT)
nbatches = int(duration / (batch_len * myconfig.DT))
N_EPOCHS = 1 #1500  # Number of training epochs


if PATH_TO_MODEL is None:
    # === MODE: Build from scratch ===
    print("="*60)
    print("MODE: Building model from scratch")
    print("="*60)
    
    # Load parameters from base model

    # optim_parameters_connection, optim_parameters_neurons = get_base_params(SOURCE_MODEL, 'local_model')
    params, generators_params, target_params, output_masks = get_params(sheet_name='full_local_model')

    Xtrain, Ytrain = get_dataset(target_params, myconfig.DT, batch_len, nbatches, output_masks)

    print("Xtrain shape:", Xtrain.shape)
    print("Ytrain shape:", Ytrain.shape)

    # Build model from scratch
    model = get_model(params, generators_params, myconfig.DT, output_masks, path_to_model=None)

    # # Save the initial model
    # model.save(myconfig.OUTPUTSPATH_MODELS + OUTPUT_MODEL_NAME)
    # print(f"Model saved to: {myconfig.OUTPUTSPATH_MODELS + OUTPUT_MODEL_NAME}")
    #
    # Setup callbacks for training
    checkpoint_filepath = myconfig.OUTPUTSPATH_MODELS + 'pyr_only_model_{epoch:02d}.keras'

    callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath,
                        save_weights_only=False,
                        monitor='loss',
                        mode='auto',
                        save_best_only=False,
                        save_freq='epoch'),
        TerminateOnNaN(),
    ]
    #
    # # Train model
    # # Note: Xtrain is time only. SpatialThetaGenerators produces all firing rates internally.
    # # MeanFieldNetwork(use_input=True) receives generator+interneuron firing rates as external input.
    history = model.fit(x=Xtrain, y=Ytrain, epochs=N_EPOCHS, verbose=2, batch_size=myconfig.BATCH_SIZE, callbacks=callbacks)
    #
    # # Save training history
    with h5py.File(myconfig.OUTPUTSPATH + 'pyr_only_history.h5', mode='w') as dfile:
        dfile.create_dataset('loss', data=history.history['loss'])
    
    print("Training complete!")

else:
    # === MODE: Fine-tune from existing model ===
    print("="*60)
    print(f"MODE: Fine-tuning from {PATH_TO_MODEL}")
    print("="*60)
    
    # Load model (don't need to rebuild params/dataset for loading)
    model = get_model(None, None, myconfig.DT, None, path_to_model=PATH_TO_MODEL)
    
    print("Model loaded successfully. Ready for fine-tuning.")
    print("To fine-tune, uncomment the training code below and provide Xtrain, Ytrain.")
    
    # # For fine-tuning, you still need to generate Xtrain, Ytrain
    # SOURCE_MODEL = PATH_TO_MODEL  # Use the loaded model as base
    # optim_parameters_connection, optim_parameters_neurons = get_base_params(SOURCE_MODEL, 'local_model')
    # params, generators_params, target_params, output_masks = get_params(optim_parameters_connection, optim_parameters_neurons, sheet_name='full_local_model')
    # Xtrain, Ytrain = get_dataset(target_params, myconfig.DT, batch_len, nbatches, output_masks)
    #
    # # Rebuild model with loaded weights
    # model = get_model(params, generators_params, myconfig.DT, output_masks, path_to_model=PATH_TO_MODEL)
    #
    # checkpoint_filepath = myconfig.OUTPUTSPATH_MODELS + 'fine_tuned_model_{epoch:02d}.keras'
    # callbacks = [
    #     ModelCheckpoint(filepath=checkpoint_filepath,
    #                     save_weights_only=False,
    #                     monitor='loss',
    #                     mode='auto',
    #                     save_best_only=False,
    #                     save_freq='epoch'),
    #     TerminateOnNaN(),
    # ]
    #
    # history = model.fit(x=Xtrain, y=Ytrain['full_output'], epochs=N_EPOCHS, verbose=2, batch_size=1, callbacks=callbacks)
    #
    # with h5py.File(myconfig.OUTPUTSPATH + 'fine_tuned_history.h5', mode='w') as dfile:
    #     dfile.create_dataset('loss', data=history.history['loss'])

