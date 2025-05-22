import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import h5py
import izhs_lib
from main import get_dataset
from pprint import pprint
import sys
sys.stderr = open('err.txt', 'w')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RNN, Layer
from mean_field_class import MeanFieldNetwork, SaveFirings
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN


from genloss import SpatialThetaGenerators, CommonOutProcessing, PhaseLockingOutputWithPhase, PhaseLockingOutput, RobastMeanOut, FiringsMeanOutRanger, Decorrelator
import myconfig


def get_params_from_pop_conns(populations, connections, neurons_params, synapses_params, dt_dim, Delta_eta):

    params = {}

    NN = 0
    Ninps =  0
    for pop_idx, pop in enumerate(populations):
        if pop["type"].find("generator") == -1:
            NN += 1
        else:
            Ninps += 1

    dimpopparams = {
        'dt_dim' : dt_dim,
        'Delta_eta' : Delta_eta,
        'I_ext' : [],
    }

    generators_params = []

    simple_out_mask = np.zeros(NN, dtype='bool')
    frequecy_filter_out_mask = np.zeros(NN, dtype='bool')
    phase_locking_out_mask = np.zeros(NN, dtype='bool')
    LowFiringRateBound = []
    HighFiringRateBound = []

    for pop_idx, pop in enumerate(populations):
        pop_type = pop['type']


        if 'generator' in pop_type:
            generators_params.append(pop)
            continue

        try:
            LowFiringRateBound.append(pop["MinFiringRate"])
            HighFiringRateBound.append(pop["MaxFiringRate"])
        except KeyError:
            pass


        if pop_type == "CA1 Pyramidal":
            simple_out_mask[pop_idx] = True
        else:
            try:
                if np.isnan(pop["ThetaPhase"]) or (pop["ThetaPhase"] is None):
                    phase_locking_out_mask[pop_idx] = True
                else:
                    frequecy_filter_out_mask[pop_idx] = True
            except KeyError:
                continue



        p = neurons_params[neurons_params["Neuron Type"] == pop_type]

        try:
            dimpopparams['I_ext'].append(pop['I_ext'])
        except KeyError:
            dimpopparams['I_ext'].append(0.0)


        for key in p:
            val = p[key].values[0]
            try:
                val  = float(val)
            except ValueError:
                continue

            if key in dimpopparams.keys():
                dimpopparams[key].append(val)
            else:
                dimpopparams[key] = [val, ]

    for key, val in dimpopparams.items():
        dimpopparams[key] = np.asarray(val)




    gsyn_max = np.zeros(shape=(NN + Ninps, NN), dtype=np.float32)

    dimpopparams['gsyn_max'] = gsyn_max
    dimpopparams["Erev"] = np.zeros_like(gsyn_max) - 75.0

    params['pconn'] = np.zeros_like(gsyn_max)
    params['tau_d'] = np.zeros_like(gsyn_max) + 100 #+ tau_d
    params['tau_r'] = np.zeros_like(gsyn_max) + 10 #+ tau_r
    params['tau_f'] = np.zeros_like(gsyn_max) + 5 #+ tau_f
    params['Uinc'] = np.zeros_like(gsyn_max) + 0.5 #+ Uinc

    for conn in connections:
        pre_idx = conn['pre_idx']
        post_idx = conn['post_idx']

        params['pconn'][pre_idx, post_idx] = conn['pconn']

        pre_type = conn['pre_type']

        if pre_type == "CA3_generator":
            pre_type = 'CA3 Pyramidal'

        if pre_type == "CA1 Pyramidal_generator":
            pre_type = 'CA1 Pyramidal'

        if pre_type == "MEC_generator":
            pre_type = 'EC LIII Pyramidal'

        if pre_type == "LEC_generator":
            pre_type = 'EC LIII Pyramidal'

            # pre_type = pre_type.replace("_generator", "")

        syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == pre_type) & (
                synapses_params['Postsynaptic Neuron Type'] == conn['post_type'])]

        if len(syn) == 0:
            print("Connection from ", conn["pre_type"], "to", conn["post_type"], "not finded!")
            continue

        Uinc = syn['Uinc'].values[0]
        tau_r = syn['tau_r'].values[0]
        tau_f = syn['tau_f'].values[0]
        tau_d = syn['tau_d'].values[0]

        if neurons_params[neurons_params['Neuron Type'] == pre_type]['E/I'].values[0] == "e":
            Erev = 0
        elif neurons_params[neurons_params['Neuron Type'] == pre_type]['E/I'].values[0] == "i":
            Erev = -75.0
        if not conn['gsyn_max'] is None:
            gsyn_max = conn['gsyn_max']
        else:
            gsyn_max = syn['gsyn_max'].values[0]

        params['Uinc'][pre_idx, post_idx]  = Uinc
        params['tau_r'][pre_idx, post_idx] = tau_r
        params['tau_f'][pre_idx, post_idx] = tau_f
        params['tau_d'][pre_idx, post_idx] = tau_d
        dimpopparams['Erev'][pre_idx, post_idx] = Erev
        dimpopparams['gsyn_max'][pre_idx, post_idx] = gsyn_max

    params_dimless = izhs_lib.dimensional_to_dimensionless_all(dimpopparams)


    params = params | params_dimless


    # params['alpha'][:] = 0.38348082
    # params['a'][:] = 0.0083115
    # params['b'][:] = 0.00320795
    # params['w_jump'][:] = 0.00050604
    # params['Delta_eta'][:] = 0.02024164
    # params['dts_non_dim'][:] = 0.06015763
    # for key, val in params.items():
    #     print(key, "\n", val)



    LowFiringRateBound = np.asarray(LowFiringRateBound)
    HighFiringRateBound = np.asarray(HighFiringRateBound)
    return params, generators_params, LowFiringRateBound, HighFiringRateBound, simple_out_mask, frequecy_filter_out_mask, phase_locking_out_mask


def get_model():
    Delta_eta = 15
    # load data about network
    if myconfig.RUNMODE == 'DEBUG':
        neurons_path = myconfig.STRUCTURESOFNET + "test_neurons.pickle"
        connections_path = myconfig.STRUCTURESOFNET + "test_conns.pickle"
    else:
        neurons_path = myconfig.STRUCTURESOFNET + "neurons.pickle"
        connections_path = myconfig.STRUCTURESOFNET + "connections.pickle"

    with open(neurons_path, "rb") as neurons_file:
        populations = pickle.load(neurons_file)

    with open(connections_path, "rb") as synapses_file:
        connections = pickle.load(synapses_file)

    neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
    neurons_params.rename(
        {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

    synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
    synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)

    Xtrain, Ytrain = get_dataset(populations)

    params, generators_params, LowFiringRateBound, HighFiringRateBound, simple_out_mask, frequecy_filter_out_mask, phase_locking_out_mask = get_params_from_pop_conns(
        populations, connections, neurons_params, synapses_params, dt_dim, Delta_eta)

    input = Input(shape=(None, 1), batch_size=1)

    generators = SpatialThetaGenerators(generators_params)(input)
    net_layer = RNN(MeanFieldNetwork(params, dt_dim=dt_dim, use_input=True),
                    return_sequences=True, stateful=True,
                    # activity_regularizer=Decorrelator(strength=0.001),
                    name="firings_outputs")(generators)

    # output_layers = []
    #
    # simple_selector = CommonOutProcessing(simple_out_mask, name='pyramilad_mask')
    # output_layers.append(simple_selector(net_layer))
    #
    # theta_phase_locking_with_phase = PhaseLockingOutputWithPhase(mask=frequecy_filter_out_mask, \
    #                                                              ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT,
    #                                                              name='locking_with_phase')
    # output_layers.append(theta_phase_locking_with_phase(net_layer))
    #
    # robast_mean_out = RobastMeanOut(mask=frequecy_filter_out_mask, name='robast_mean')
    # output_layers.append(robast_mean_out(net_layer))
    #
    # phase_locking_selector = PhaseLockingOutput(mask=phase_locking_out_mask,
    #                                             ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT, name='locking')
    # output_layers.append(phase_locking_selector(net_layer))

    # net_layer = Layer(activity_regularizer=FiringsMeanOutRanger(LowFiringRateBound=LowFiringRateBound, HighFiringRateBound=HighFiringRateBound))(net_layer)

    outputs = net_layer # generators #
    big_model = Model(inputs=input, outputs=outputs)



    big_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=myconfig.LEARNING_RATE, clipvalue=0.1),
        loss = tf.keras.losses.logcosh
        # loss={
        #     'pyramilad_mask': tf.keras.losses.logcosh,
        #     'locking_with_phase': tf.keras.losses.logcosh,
        #     'robast_mean': tf.keras.losses.logcosh,
        #     'locking': tf.keras.losses.logcosh,
        # }
    )

    #big_model = tf.keras.models.clone_model(big_model)



    return big_model, Xtrain, Ytrain
##################################################################
if __name__ == '__main__':
    dt_dim = myconfig.DT

    duration_full_simulation = 1000 * myconfig.TRACK_LENGTH / myconfig.ANIMAL_VELOCITY  # ms
    t_full = np.arange(0, duration_full_simulation, myconfig.DT).reshape(1, -1, 1)

    big_model, Xtrain, Ytrain = get_model()

    firings_outputs_layer = big_model.get_layer('firings_outputs').output
    firings_model = Model(inputs=big_model.input, outputs=firings_outputs_layer)

    checkpoint_filepath = myconfig.OUTPUTSPATH_MODELS + 'big_model_{epoch:02d}.keras'
    filename_template = 'firings_{epoch:02d}.h5'

    Nepoches4modelsaving = 2 * len(Xtrain) + 1

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1)

    callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='loss',
            mode='auto',
            save_best_only=False,
            save_freq = 'epoch'),

        SaveFirings( firing_model=firings_model,
                     t_full=t_full,
                     path=myconfig.OUTPUTSPATH_FIRINGS,
                     filename_template=filename_template,
                     save_freq = 1),
       #  tensorboard_callback,
       TerminateOnNaN(),
    ]


    # del Ytrain['robast_mean']
    # del Ytrain['locking']

    # for key, val in Ytrain.items():
    #      print( key, val.shape )
    history = big_model.fit(x=Xtrain, y=Ytrain, epochs=20, verbose=2, batch_size=1, callbacks=callbacks)
    # loss = big_model.evaluate(x=Xtrain, y=Ytrain, verbose=2, batch_size=1)
    # pprint(loss)

    # Ys = big_model.predict(Xtrain, batch_size=1)




    # for y_idx, (ypred, ytrain) in enumerate(zip(Ys, Ytrain.values())):


    #     print('N of nans', np.sum( np.isnan(ypred) ) )
    #     print('##############################')

    #
    #
    #     if y_idx == 0:
    #         with h5py.File('./outputs/firings/pyr_firings.h5', 'w') as h5file:
    #             h5file.create_dataset('firings', data=y)
    #
    #
    #         import matplotlib.pyplot as plt
    #         #y = y.reshape()
    #         plt.plot(y[0, :, 0] )
    #         plt.show()




