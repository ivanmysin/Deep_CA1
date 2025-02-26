import os

import matplotlib.pyplot as plt

os.chdir('../')

import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import h5py

from pprint import pprint
import myconfig

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RNN, Reshape
from tensorflow.keras.saving import load_model
from genloss import SpatialThetaGenerators, CommonOutProcessing, PhaseLockingOutputWithPhase, PhaseLockingOutput, RobastMeanOut, FiringsMeanOutRanger, Decorrelator
from time_step_layer import TimeStepLayer

def get_model(populations, connections, neurons_params, synapses_params, base_pop_models):

    spatial_gen_params = []
    Ns = 0
    for pop_idx, pop in enumerate(populations):
        if pop["type"].find("generator") == -1:
            Ns += 1
        else:
            spatial_gen_params.append(pop)

    simple_out_mask = np.zeros(Ns, dtype='bool')
    frequecy_filter_out_mask = np.zeros(Ns, dtype='bool')
    phase_locking_out_mask = np.zeros(Ns, dtype='bool')
    LowFiringRateBound = []
    HighFiringRateBound = []

    for pop_idx, pop in enumerate(populations):

        try:
            LowFiringRateBound.append(pop["MinFiringRate"])
            HighFiringRateBound.append(pop["MaxFiringRate"])
        except KeyError:
            if not 'generator' in pop['type']:
                print(pop['type'], "No MinFiringRate or MaxFiringRate")
            continue

        if pop["type"] == "CA1 Pyramidal":
            simple_out_mask[pop_idx] = True

        else:
            try:
                if np.isnan(pop["ThetaPhase"]) or (pop["ThetaPhase"] is None):
                    phase_locking_out_mask[pop_idx] = True
                else:
                    frequecy_filter_out_mask[pop_idx] = True
            except KeyError:
                continue


    input = Input(shape=(None, 1), batch_size=1)
    generators = SpatialThetaGenerators(spatial_gen_params) #(input)

    time_step_layer = TimeStepLayer(Ns, populations, connections, neurons_params, synapses_params, base_pop_models, dt=myconfig.DT)
    time_step_layer_rnn = RNN(time_step_layer, return_sequences=True, stateful=True)

    time_step_layer_rnn = time_step_layer_rnn(generators(input))

    time_step_layer_rnn = Reshape(target_shape=(-1, Ns), name="firings_outputs")(time_step_layer_rnn)


    firings_model = Model(inputs=input, outputs=time_step_layer_rnn)

    return firings_model, time_step_layer, generators

def main():
    # load data about network
    if myconfig.RUNMODE == 'DEBUG':
        neurons_path = myconfig.STRUCTURESOFNET + "test_neurons.pickle"
        connections_path = myconfig.STRUCTURESOFNET + "test_conns.pickle"
    else:
        neurons_path = myconfig.STRUCTURESOFNET + "neurons.pickle"
        connections_path = myconfig.STRUCTURESOFNET + "connections.pickle"


    with open(neurons_path, "rb") as neurons_file: ##!!
        populations = pickle.load(neurons_file)

    with open(connections_path, "rb") as synapses_file: ##!!
        connections = pickle.load(synapses_file)


    pop_types_params = pd.read_excel(myconfig.SCRIPTS4PARAMSGENERATION + "neurons_parameters.xlsx", sheet_name="Sheet2",
                                     header=0)


    neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
    neurons_params.rename(
        {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)
    synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
    synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)

    base_pop_models = {}
    for pop_idx, population in pop_types_params.iterrows():
        if not population["is_include"]:
            continue

        pop_type = population["neurons"]
        base_pop_models[pop_type] = myconfig.PRETRANEDMODELS + pop_type + '.keras'


    firings_model, time_step_layer, generators = get_model(populations, connections, neurons_params, synapses_params, base_pop_models)

    duration_full_simulation = 5000 # 1000 * myconfig.TRACK_LENGTH / myconfig.ANIMAL_VELOCITY # ms
    t = np.arange(0, duration_full_simulation, myconfig.DT).reshape(1, -1, 1)



    firings = firings_model.predict(t)
    firings_generators = generators(t)
    firings_generators = firings_generators.numpy()
    firings = np.append(firings, firings_generators, axis=2)


    syn_pop_model = time_step_layer.pop_models[2]
    external_inputs = np.zeros(shape=(1, t.size, len(populations)), dtype=np.float32)
    external_inputs[0, :, 5:] = firings_generators[0, :, :]

    pv_bas_firing = syn_pop_model.predict(external_inputs)
    #pv_bas_firing = []
    # for t_idx in range(t.size):
    #     external_inputs_tmp = external_inputs[:, t_idx, :]
    #     external_inputs_tmp = external_inputs_tmp.reshape(1, 1, -1)
    #     pv_bas_firing_tmp = syn_pop_model.predict(external_inputs_tmp)
    #
    #     pv_bas_firing.append(pv_bas_firing_tmp)
    #
    # pv_bas_firing = np.stack(pv_bas_firing, axis=1)
    # print('pv_bas_firing.shape:', pv_bas_firing.shape)

    nsubplots = firings.shape[-1]



    # with h5py.File("firings.h5", mode='w') as h5file:
    #     h5file.create_dataset('firings', data=firings)


    t = t.ravel()
    fig, axes = plt.subplots(nrows=nsubplots, sharex=True, sharey=False)
    for f_idx in range(nsubplots):
        axes[f_idx].set_title(populations[f_idx]['type'])
        axes[f_idx].plot(t, firings[0, :, f_idx], linewidth=5)

    axes[2].plot(t, pv_bas_firing.ravel(), linewidth=1)

    plt.show()











##########################################################################
if __name__ == '__main__':
    main()