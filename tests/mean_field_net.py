import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import h5py
import izhs_lib
from pprint import pprint


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RNN, Layer
from test_mean_field_class import MeanFieldNetwork, SaveFirings
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import os
os.chdir('../')
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
    dimpopparams["Erev"] = np.zeros_like(gsyn_max)

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

        params['Uinc'][pre_idx, post_idx]  = Uinc
        params['tau_r'][pre_idx, post_idx]  = tau_r
        params['tau_f'][pre_idx, post_idx] = tau_f
        params['tau_d'][pre_idx, post_idx]  = tau_d
        dimpopparams['Erev'][pre_idx, post_idx] = Erev
        dimpopparams['gsyn_max'][pre_idx, post_idx] = conn['gsyn_max']

    params_dimless = izhs_lib.dimensional_to_dimensionless_all(dimpopparams)


    params = params | params_dimless

    LowFiringRateBound = np.asarray(LowFiringRateBound)
    HighFiringRateBound = np.asarray(HighFiringRateBound)
    return params, generators_params, LowFiringRateBound, HighFiringRateBound, simple_out_mask, frequecy_filter_out_mask, phase_locking_out_mask




##################################################################
dt_dim = myconfig.DT
Delta_eta = 80
duration = 1000


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


params, generators_params, LowFiringRateBound, HighFiringRateBound, simple_out_mask, frequecy_filter_out_mask, phase_locking_out_mask = get_params_from_pop_conns(populations, connections, neurons_params, synapses_params, dt_dim, Delta_eta)


print(params['pconn'])
print(params['gsyn_max'])
print(params['tau_f'])

input = Input(shape=(None, 1), batch_size=1)
generators = SpatialThetaGenerators(generators_params)(input)
net_layer = RNN( MeanFieldNetwork(params, dt_dim=dt_dim, use_input=True),
                 return_sequences=True, stateful=True,
                 activity_regularizer=Decorrelator(strength=0.001),
                 name="firings_outputs")(generators)

output_layers = []

simple_selector = CommonOutProcessing(simple_out_mask, name='pyramilad_mask')
output_layers.append(simple_selector(net_layer))

theta_phase_locking_with_phase = PhaseLockingOutputWithPhase(mask=frequecy_filter_out_mask, \
                                                                 ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT,
                                                                 name='locking_with_phase')
output_layers.append(theta_phase_locking_with_phase(net_layer))

robast_mean_out = RobastMeanOut(mask=frequecy_filter_out_mask, name='robast_mean')
output_layers.append(robast_mean_out(net_layer))

phase_locking_selector = PhaseLockingOutput(mask=phase_locking_out_mask,
                                                ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT, name='locking')
output_layers.append(phase_locking_selector(net_layer))



#net_layer = Layer(activity_regularizer=FiringsMeanOutRanger(LowFiringRateBound=LowFiringRateBound, HighFiringRateBound=HighFiringRateBound))(net_layer)


outputs = output_layers #net_layer # generators #
big_model = Model(inputs=input, outputs=outputs)

big_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=myconfig.LEARNING_RATE, clipvalue=0.1),
    # loss = tf.keras.losses.logcosh
    loss={
        'pyramilad_mask': tf.keras.losses.logcosh,
        'locking_with_phase': tf.keras.losses.MSE,
        'robast_mean': tf.keras.losses.MSE,
        'locking': tf.keras.losses.MSE,
    }
)

t = tf.range(0, duration, dt_dim, dtype=tf.float32)
t = tf.reshape(t, shape=(1, -1, 1))

#y = big_model.predict(t)

firings_outputs_layer = big_model.get_layer('firings_outputs').output
firings_model = Model(inputs=big_model.input, outputs=firings_outputs_layer)

firings = firings_model.predict(t)
with h5py.File(myconfig.OUTPUTSPATH_FIRINGS + f'test_firings.h5', mode='w') as h5file:
    h5file.create_dataset('firings', data=firings)



