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


def save_trained_to_pickle(trainable_variables, connections):

    for tv in trainable_variables:
        pop_idx = int(tv.name.split("_")[-1])

        tv = tv.numpy()

        conn_counter = 0
        for conn in connections:
            if conn["post_idx"] != pop_idx:
                continue

            conn["gsyn"] = tv[conn_counter]

            conn_counter += 1

    if myconfig.RUNMODE == 'DEBUG':
        saving_path = "./presimulation_files/test_conns.pickle"
    else:
        saving_path = myconfig.STRUCTURESOFNET + "connections.pickle"

    with open(saving_path, mode="bw") as file:
        pickle.dump(connections, file)



def get_dataset(populations):
    dt = myconfig.DT
    duration_full_simulation = 1000 * myconfig.TRACK_LENGTH / myconfig.ANIMAL_VELOCITY # ms
    full_time_steps = duration_full_simulation / dt

    n_times_batches = int(np.floor(full_time_steps / myconfig.N_TIMESTEPS))

    pyramidal_targets = []
    phase_locking_with_phase = []
    phase_locking_without_phase = []
    robast_mean_firing_rate = []

    for pop_idx, pop in enumerate(populations):
        if pop["type"] == "CA1 Pyramidal":
            pyramidal_targets.append(pop)

        else:
            try:
                if np.isnan(pop["ThetaPhase"]) or (pop["ThetaPhase"] is None):
                    phase_locking_without_phase.append(pop["R"])

                else:
                    phase_locking_with_phase.append([pop["ThetaPhase"], pop["R"]])
                    robast_mean_firing_rate.append(pop["MeanFiringRate"])

            except KeyError:
                continue

    phase_locking_without_phase = np.asarray(phase_locking_without_phase).reshape(1, 1, -1)
    robast_mean_firing_rate = np.asarray(robast_mean_firing_rate).reshape(1, 1, -1)

    phase_locking_with_phase = np.asarray(phase_locking_with_phase)

    im = phase_locking_with_phase[:, 1] * np.sin(phase_locking_with_phase[:, 0])
    re = phase_locking_with_phase[:, 1] * np.cos(phase_locking_with_phase[:, 0])

    phase_locking_with_phase = np.stack([re, im], axis=1).reshape(1, 2, -1)

    generators = SpatialThetaGenerators(pyramidal_targets)

    Xtrain = []
    Ytrain = []

    t0 = 0.0
    for batch_idx in range(n_times_batches):
        tend = t0 + myconfig.N_TIMESTEPS * dt
        t = np.arange(t0, tend, dt).reshape(1, -1, 1)
        t0 = tend

        Xtrain.append(t)

        pyr_targets = generators(t)

        # print(pyr_targets.shape)
        # print(phase_locking_with_phase.shape)
        # print(robast_mean_firing_rate.shape)
        # print(phase_locking_without_phase.shape)

        Ytrain.append({
            'pyramilad_mask': pyr_targets,
            'locking_with_phase': np.copy(phase_locking_with_phase),
            'robast_mean': np.copy(robast_mean_firing_rate),
            'locking' : np.copy(phase_locking_without_phase),
        })

        # if batch_idx == 0:
        #     print(pyr_targets.shape)
        #     print(phase_locking_with_phase.shape)
        #     print(robast_mean_firing_rate.shape)
        #     print(phase_locking_without_phase.shape)



    return Xtrain, Ytrain


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


    # print("Pyramidal", np.sum(simple_out_mask))
    # print("frequecy_filter_out_mask", np.sum(frequecy_filter_out_mask))
    # print("phase_locking_out_mask", np.sum(phase_locking_out_mask))


    input = Input(shape=(None, 1), batch_size=1)
    generators = SpatialThetaGenerators(spatial_gen_params)(input)

    time_step_layer = TimeStepLayer(Ns, populations, connections, neurons_params, synapses_params, base_pop_models, dt=myconfig.DT)
    time_step_layer = RNN(time_step_layer, return_sequences=True, stateful=True,
                          activity_regularizer=FiringsMeanOutRanger(LowFiringRateBound=LowFiringRateBound, HighFiringRateBound=HighFiringRateBound))

    time_step_layer = time_step_layer(generators)

    time_step_layer = Reshape(target_shape=(-1, Ns), activity_regularizer=Decorrelator(strength=0.1), name="firings_outputs")(time_step_layer)

    output_layers = []

    simple_selector = CommonOutProcessing(simple_out_mask, name='pyramilad_mask')
    output_layers.append(simple_selector(time_step_layer))

    theta_phase_locking_with_phase = PhaseLockingOutputWithPhase(mask=frequecy_filter_out_mask, \
                                                                 ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT,
                                                                 name='locking_with_phase')
    output_layers.append(theta_phase_locking_with_phase(time_step_layer))

    robast_mean_out = RobastMeanOut(mask=frequecy_filter_out_mask, name='robast_mean')
    output_layers.append(robast_mean_out(time_step_layer))

    phase_locking_selector = PhaseLockingOutput(mask=phase_locking_out_mask,
                                                ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT, name='locking')
    output_layers.append(phase_locking_selector(time_step_layer))

    #firings_model = Model(inputs=input, outputs=time_step_layer)
    big_model = Model(inputs=input, outputs=output_layers)

    # big_model.build(input_shape = (None, 1))

    big_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            'pyramilad_mask': tf.keras.losses.logcosh,
            'locking_with_phase': tf.keras.losses.MSE,
            'robast_mean': tf.keras.losses.MSE,
            'locking': tf.keras.losses.MSE,
        }
    )

    return big_model #, firings_model

def get_firings_model(bigmodel):
    firings_outputs_layer = bigmodel.get_layer('firings_outputs').output
    new_model = Model(inputs=bigmodel.input, outputs=firings_outputs_layer)
    return new_model



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

    Xtrain, Ytrain = get_dataset(populations)


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
        # if myconfig.RUNMODE == 'DEBUG':
        #     base_pop_models[pop_type] = myconfig.PRETRANEDMODELS + 'NO_Trained.keras'

    model = get_model(populations, connections, neurons_params, synapses_params, base_pop_models)
    print(model.summary())

    #model.save('big_model.keras')

    # Ypreds = model.predict(Xtrain[0])
    #
    # for y in Ypreds:
    #     print(y.shape)

    # custom_objects = {
    #     'FiringsMeanOutRanger': FiringsMeanOutRanger,
    #     'Decorrelator' : Decorrelator,
    # }
    # model = load_model('big_model.keras',  custom_objects = custom_objects)


    with tf.device('/gpu:0'):
        counter = 1
        for x_train, y_train in zip(Xtrain, Ytrain):
            #model.fit(x_train, y_train, epochs=myconfig.EPOCHES_ON_BATCH, verbose=2)
            model.train_on_batch(x_train, y_train)
            #y_tmp = model.predict(x_train)
            model.save('big_model.keras')
            print(counter)
            counter += 1
    #save_trained_to_pickle(model.trainable_variables, connections)



    # firings_model = get_firings_model(model)
    #
    # duration_full_simulation = 1000 * myconfig.TRACK_LENGTH / myconfig.ANIMAL_VELOCITY # ms
    # t = np.arange(0, duration_full_simulation, myconfig.DT).reshape(1, -1, 1)
    # firings = firings_model.predict(t)
    # with h5py.File("firings.h5", mode='w') as h5file:
    #     h5file.create_dataset('firings', data=firings)




##########################################################################
if __name__ == '__main__':
    main()