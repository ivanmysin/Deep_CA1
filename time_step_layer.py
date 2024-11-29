import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, RNN, Layer, Reshape
import warnings
import myconfig

from synapses_layers import TsodycsMarkramSynapse
from genloss import SpatialThetaGenerators, CommonOutProcessing, PhaseLockingOutputWithPhase, PhaseLockingOutput, RobastMeanOut, RobastMeanOutRanger, Decorrelator


class TimeStepLayer(Layer):

    def __init__(self, units, populations, connections, neurons_params, synapses_params, base_pop_models, dt=0.1,  **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.dt = dt
        self.input_size = len(populations)


        self.pop_models = []
        for pop_idx, pop in enumerate(populations):
            if "_generator" in pop["type"]:
                continue


            base_model = base_pop_models[pop["type"]]
            pop_model = self.get_model(pop_idx, pop, connections, base_model, neurons_params, synapses_params)
            self.pop_models.append(pop_model)





    def get_model(self, pop_idx, pop, connections, base_model, neurons_params, synapses_params):

        pop_type = pop["type"]

        if len( neurons_params[neurons_params["Neuron Type"] == pop_type] ) == 0:
            print("No neuron type", pop_type, " index: ", pop_idx)

            return None

        conn_params = {
            "gsyn_max": [],
            "Uinc": [],
            "tau_r": [],
            "tau_f": [],
            "tau_d": [],
            'pconn': [],
            'Erev': [],
            'Cm': neurons_params[neurons_params["Neuron Type"] == pop_type]["Cm"].values[0],
            'Erev_min': -75.0,
            'Erev_max': 0.0,
        }

        is_connected_mask = np.zeros(self.input_size, dtype='bool')


        #print(pop_idx)

        for conn in connections:
            if conn["post_idx"] != pop_idx: continue

            pre_type = conn['pre_type']

            if "_generator" in pre_type:
                pre_type = pre_type.replace("_generator", "")


            syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == pre_type) & (
                    synapses_params['Postsynaptic Neuron Type'] == conn['post_type'])]

            if len(syn) == 0:
                print("Connection from ", conn["pre_type"], "to", conn["post_type"], "not finded!")
                continue

            is_connected_mask[conn["pre_idx"]] = True

            conn_params["gsyn_max"].append(0.8)  # np.random.rand()!!!
            conn_params['pconn'].append(conn['pconn'])

            Uinc = syn['Uinc'].values[0]
            tau_r = syn['tau_r'].values[0]
            tau_f = syn['tau_f'].values[0]
            tau_d = syn['tau_d'].values[0]

            if neurons_params[neurons_params['Neuron Type'] == pre_type]['E/I'].values[0] == "e":
                Erev = 0
            elif neurons_params[neurons_params['Neuron Type'] == pre_type]['E/I'].values[0] == "i":
                Erev = -75.0

            conn_params['Uinc'].append(Uinc)
            conn_params['tau_r'].append(tau_r)
            conn_params['tau_f'].append(tau_f)
            conn_params['tau_d'].append(tau_d)
            conn_params['Erev'].append(Erev)



        if np.sum(is_connected_mask) == 0:
            warns_message = "No presynaptic population " + pop["type"] + " with index " + str(pop_idx)
            warnings.warn(warns_message)

        synapses = TsodycsMarkramSynapse(conn_params, dt=self.dt, mask=is_connected_mask)

        print("pop_idx: ", pop_idx)
        synapses_layer = RNN(synapses, return_sequences=True, stateful=True, name=f"Synapses_Layer_Pop_{pop_idx}")

        input_layer = Input(shape=(None, self.input_size), batch_size=1)
        synapses_layer = synapses_layer(input_layer)

        base_model = tf.keras.models.clone_model(base_model)

        model = Model(inputs=input_layer, outputs=base_model(synapses_layer), name="Population_with_synapses")


        return tf.keras.models.clone_model(model)



    def get_initial_state(self, batch_size=1):
        state = K.zeros(shape=(batch_size, self.state_size), dtype=tf.float32)
        return state

    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.n_dims = input_shape[-1]


    def call(self, input, state):


        input = K.concatenate([state[0], input], axis=-1)
        input = K.reshape(input, shape=(1, 1, -1))

        output = []
        for model in self.pop_models:
            out = model(input)
            output.append(out)
        output = K.concatenate(output, axis=-1)

        return output, output[0]