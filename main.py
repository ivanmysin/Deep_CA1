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
from tensorflow.keras.callbacks import ModelCheckpoint


def save_trained_to_pickle(trainable_variables, connections):

    for tv in trainable_variables:
        pop_idx = int(tv.name.split("_")[-1])

        tv = tv.numpy()

        conn_counter = 0
        for conn in connections:
            if conn["post_idx"] != pop_idx:
                continue

            try:
                conn["gsyn"] = tv[conn_counter]
            except IndexError:
                conn["gsyn"] = None
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
    Ytrain = {
        'pyramilad_mask': [],
        'locking_with_phase': [],
        'robast_mean': [],
        'locking': [],
    }

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

        Ytrain['pyramilad_mask'].append(pyr_targets)
        Ytrain['locking_with_phase'].append( np.copy(phase_locking_with_phase) )
        Ytrain['robast_mean'].append( np.copy(robast_mean_firing_rate) )
        Ytrain['locking'].append( np.copy(phase_locking_without_phase) )


    Xtrain = np.concatenate(Xtrain, axis=0)
    for key, val in Ytrain.items():
        Ytrain[key] = np.concatenate(val, axis=0)


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
    time_step_layer = RNN(time_step_layer, return_sequences=True, stateful=True) #, activity_regularizer=FiringsMeanOutRanger(LowFiringRateBound=LowFiringRateBound, HighFiringRateBound=HighFiringRateBound))

    time_step_layer = time_step_layer(generators)

    time_step_layer = Reshape(target_shape=(-1, Ns), activity_regularizer=Decorrelator(strength=0.001), name="firings_outputs")(time_step_layer)
    ### time_step_layer = Reshape(target_shape=(-1, Ns), name="firings_outputs")(time_step_layer)

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
        optimizer = tf.keras.optimizers.Adam(learning_rate=myconfig.LEARNING_RATE, clipvalue=0.1),
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


    with open(neurons_path, "rb") as neurons_file:
        populations = pickle.load(neurons_file)

    with open(connections_path, "rb") as synapses_file:
        connections = pickle.load(synapses_file)

        # for conn in connections:
        #     conn['pconn'] *= 100

    Xtrain, Ytrain = get_dataset(populations)


    pop_types_params = pd.read_excel(myconfig.SCRIPTS4PARAMSGENERATION + "neurons_parameters.xlsx", sheet_name="Sheet2",
                                     header=0)


    neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
    neurons_params.rename(
        {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

    neurons_params['Cm'] *= 0.001 # recalculate pF to nF
    neurons_params['k'] *= 0.001 # recalculate nS to pS

    synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
    synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)

    synapses_params["gsyn_max"] *= 0.1 # !!!!

    base_pop_models = {}
    for pop_idx, population in pop_types_params.iterrows():
        if not population["is_include"]:
            continue

        pop_type = population["neurons"]
        base_pop_models[pop_type] = myconfig.PRETRANEDMODELS + pop_type + '.keras'
        # if myconfig.RUNMODE == 'DEBUG':
        #     base_pop_models[pop_type] = myconfig.PRETRANEDMODELS + 'NO_Trained.keras'

    model = get_model(populations, connections, neurons_params, synapses_params, base_pop_models)
    firings_model = get_firings_model(model)
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

    duration_full_simulation = 1000 * myconfig.TRACK_LENGTH / myconfig.ANIMAL_VELOCITY  # ms
    t_full = np.arange(0, duration_full_simulation, myconfig.DT).reshape(1, -1, 1)

    # fname = "weights-{epoch:03d}-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(fname, monitor="loss", mode="min",
    #                              period=10, verbose=1)
    # callbacks = [checkpoint, ]


    with tf.device('/gpu:0'):
        #loss_hist = []
        for epoch_idx in range(myconfig.EPOCHES_FULL_T):

            history = model.fit(Xtrain, Ytrain, epochs=myconfig.EPOCHES_ON_BATCH, verbose=2, batch_size=1)
            loss = history.history['loss']
            #loss = model.train_on_batch(x_train, y_train)


            epoch_counter = epoch_idx + 1
            model.save(myconfig.OUTPUTSPATH_MODELS + f'{epoch_counter}_big_model.keras')
            save_trained_to_pickle(model.trainable_variables, connections)

            firings = firings_model.predict(t_full)
            with h5py.File(myconfig.OUTPUTSPATH_FIRINGS + f'{epoch_counter}_firings.h5', mode='w') as h5file:
                h5file.create_dataset('firings', data=firings)
                h5file.create_dataset('loss_hist', data=np.asarray(loss))

            print("Full time epoches", epoch_counter)
            #print("Loss over epoche", loss_hist[-1])








##########################################################################
if __name__ == '__main__':
    main()