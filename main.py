import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from pprint import pprint
import myconfig

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RNN, Reshape
from tensorflow.keras.saving import load_model
from synapses_layers import TsodycsMarkramSynapse
from genloss import SpatialThetaGenerators, CommonOutProcessing, PhaseLockingOutputWithPhase, PhaseLockingOutput, RobastMeanOut, RobastMeanOutRanger, Decorrelator
from time_step_layer import TimeStepLayer


def get_model(populations, connections, neurons_params, synapses_params, base_pop_models):

    for base_model in base_pop_models.values():
        for layer in base_model.layers:
            layer.trainable = False

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
    for pop_idx, pop in enumerate(populations):
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
    generators = SpatialThetaGenerators(spatial_gen_params)(input)

    time_step_layer = TimeStepLayer(Ns, populations, connections, neurons_params, synapses_params, base_pop_models, dt=myconfig.DT)
    time_step_layer = RNN(time_step_layer, return_sequences=True, stateful=True,
                          activity_regularizer=RobastMeanOutRanger())

    time_step_layer = time_step_layer(generators)

    time_step_layer = Reshape(target_shape=(-1, Ns), activity_regularizer=Decorrelator(strength=0.1))(time_step_layer)

    output_layers = []

    simple_selector = CommonOutProcessing(simple_out_mask, name='pyramilad_mask')
    output_layers.append(simple_selector(time_step_layer))

    theta_phase_locking_with_phase = PhaseLockingOutputWithPhase(mask=frequecy_filter_out_mask, \
                                                                 ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT,
                                                                 name='locking_with_phase')
    output_layers.append(theta_phase_locking_with_phase(time_step_layer))

    robast_mean_out = RobastMeanOut(mask=frequecy_filter_out_mask, name='robast_mean')
    output_layers.append(robast_mean_out(time_step_layer))

    phase_locking_out_mask = np.ones(Ns, dtype='bool')
    phase_locking_selector = PhaseLockingOutput(mask=phase_locking_out_mask,
                                                ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT, name='locking')
    output_layers.append(phase_locking_selector(time_step_layer))

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

    return big_model



def main():
    # load data about network
    with open(myconfig.STRUCTURESOFNET + "test_neurons.pickle", "rb") as neurons_file:
    # with open(myconfig.STRUCTURESOFNET + "neurons.pickle", "rb") as neurons_file:
        populations = pickle.load(neurons_file)

    with open(myconfig.STRUCTURESOFNET + "test_conns.pickle", "rb") as synapses_file:
    #with open(myconfig.STRUCTURESOFNET + "connections.pickle", "rb") as synapses_file:
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
    for pop_type in pop_types_params["neurons"]:
        #model_file = myconfig.PRETRANEDMODELS + pop_type + '.keras'
        model_file = "./pretrained_models/NO_Trained.keras"
        base_pop_models[pop_type] = load_model(model_file)

    model = get_model(populations, connections, neurons_params, synapses_params, base_pop_models)

    print(model.summary())


##########################################################################
main()