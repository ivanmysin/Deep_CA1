import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, RNN, Layer, Reshape
from tensorflow.keras.saving import load_model
import warnings
import myconfig

from synapses_layers import TsodycsMarkramSynapse
# from genloss import SpatialThetaGenerators, CommonOutProcessing, PhaseLockingOutputWithPhase, PhaseLockingOutput, RobastMeanOut, RobastMeanOutRanger, Decorrelator


class TimeStepLayer(Layer):

    def __init__(self, units, populations, connections, neurons_params, synapses_params, base_pop_models, dt=0.1,  **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.dt = dt
        self.input_size = len(populations)

        self.populations = populations
        self.connections = connections
        self.neurons_params = neurons_params
        self.synapses_params = synapses_params

        self.base_pop_models_files = base_pop_models


        self.pop_models = []
        for pop_idx, pop in enumerate(populations):
            if "_generator" in pop["type"]:
                continue

            base_pop_model_path = self.base_pop_models_files[pop["type"]]
            base_model = load_model(base_pop_model_path, custom_objects={'square': tf.keras.ops.square})
            for layer in base_model.layers:
                layer.trainable = False

            pop_model = self.get_model(pop_idx, pop, connections, base_model, neurons_params, synapses_params)
            self.pop_models.append(pop_model)


    def copy_layers(self, old_base_model):

        model = Sequential()
        model.add(Input(shape=(None, 1), batch_size=1))

        for layer in old_base_model.layers:
            layer_type = layer.__class__.__name__

            # print(layer_type)
            # print(layer.units)

            Layer_obj = getattr(tf.keras.layers, layer_type)

            if (layer_type == 'GRU') or (layer_type == 'LSTM'):
                model.add(Layer_obj(units=layer.units, return_sequences=True, stateful=True))
            else:
                model.add(Layer_obj(units=layer.units, activation=layer.activation))

        model.build()

        for newlayer, oldlayer in zip(model.layers, old_base_model.layers):
            newlayer.set_weights(oldlayer.get_weights())
            newlayer.trainable = False

        return model



    def get_model(self, pop_idx, pop, connections, base_model, neurons_params, synapses_params):

        pop_type = pop["type"]

        if len( neurons_params[neurons_params["Neuron Type"] == pop_type] ) == 0:
            print("No neuron type", pop_type, " index: ", pop_idx)

            return None

        Vrest = neurons_params[neurons_params["Neuron Type"] == pop_type]["Vrest"].values[0]
        Vt = neurons_params[neurons_params["Neuron Type"] == pop_type]["Vth_mean"].values[0]
        k = neurons_params[neurons_params["Neuron Type"] == pop_type]["k"].values[0]
        gl = k * (Vt - Vrest)
        conn_params = {
            "gsyn_max": [],
            "Uinc": [],
            "tau_r": [],
            "tau_f": [],
            "tau_d": [],
            'pconn': [],
            'Erev': [],
            'Cm': neurons_params[neurons_params["Neuron Type"] == pop_type]["Cm"].values[0],
            'Vrest': neurons_params[neurons_params["Neuron Type"] == pop_type]["Vrest"].values[0],
            'gl': gl,
            'Erev_min': -75.0,
            'Erev_max': 0.0,
            "pop_idx" : pop_idx,
        }

        is_connected_mask = np.zeros(self.input_size, dtype='bool')


        #print(pop_idx)

        for conn in connections:
            if conn["post_idx"] != pop_idx: continue

            if is_connected_mask[conn["pre_idx"]]:

                print("Synapse already exist!!!")
                continue


            pre_type = conn['pre_type']

            if pre_type == "CA3_generator":
                pre_type = 'CA3 Pyramidal'

            if pre_type == "CA1 Pyramidal_generator":
                pre_type = 'CA1 Pyramidal'

            if pre_type == "MEC_generator":
                pre_type = 'EC LIII Pyramidal'

            if pre_type == "LEC_generator":
                pre_type = 'EC LIII Pyramidal'

                #pre_type = pre_type.replace("_generator", "")


            syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == pre_type) & (
                    synapses_params['Postsynaptic Neuron Type'] == conn['post_type'])]

            if len(syn) == 0:
                print("Connection from ", conn["pre_type"], "to", conn["post_type"], "not finded!")
                continue

            is_connected_mask[conn["pre_idx"]] = True

            conn_params["gsyn_max"].append(conn['gsyn_max'])
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


        assert( np.sum(is_connected_mask) == len(conn_params['pconn']) )

        synapses = TsodycsMarkramSynapse(conn_params, dt=self.dt, mask=is_connected_mask)
        synapses_layer = RNN(synapses, return_sequences=True, stateful=True, name=f"Synapses_Layer_Pop_{pop_idx}")

        input_layer = Input(shape=(None, self.input_size), batch_size=1)
        synapses_layer = synapses_layer(input_layer)

        # base_model = tf.keras.models.clone_model(base_model)

        base_model = self.copy_layers(base_model)

        model = Model(inputs=input_layer, outputs=base_model(synapses_layer), name=f"Population_with_synapses_{pop_idx}")

        return model



    def get_initial_state(self, batch_size=1):
        state = K.zeros(shape=(batch_size, 1, self.state_size), dtype=tf.float32)
        return (state, )

    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.n_dims = input_shape[-1]


    def call(self, input, state):

        input = tf.reshape(input, shape=(1, 1, -1))


        input = tf.concat([state[0], input], axis=-1)
        #input = tf.reshape(input, shape=(1, 1, -1))

        output = []

        for model in self.pop_models:
            out = model(input)
            output.append(out)
        output = tf.concat(output, axis=-1)


        return output, (output, )

    def get_config(self):
        config = super().get_config()

        config.update({
            'units' : self.units,
            'dt': self.dt,
            'populations' : self.populations,
            'connections' : self.connections,
            'neurons_params' : self.neurons_params.to_dict(),
            'synapses_params' : self.synapses_params.to_dict(),
            'base_pop_models' : self.base_pop_models_files,
        })

        return config


    # Статический метод для создания экземпляра класса из конфигурации
    @classmethod
    def from_config(cls, config):
        params_dict = {
            'dt': config['dt'],
        }

        params = []
        params.append(config['units']) #= [configunits, populations, connections, neurons_params, synapses_params, base_pop_models, dt=0.1]
        params.append(config['populations']) #= [configunits, populations, connections, neurons_params, synapses_params, base_pop_models, dt=0.1]
        params.append(config['connections']) #= [configunits, populations, connections, neurons_params, synapses_params, base_pop_models, dt=0.1]
        params.append( pd.DataFrame(config['neurons_params']) ) #= [configunits, populations, connections, neurons_params, synapses_params, base_pop_models, dt=0.1]
        params.append( pd.DataFrame(config['synapses_params']) ) #= [configunits, populations, connections, neurons_params, synapses_params, base_pop_models, dt=0.1]
        params.append(config['base_pop_models']) #= [configunits, populations, connections, neurons_params, synapses_params, base_pop_models, dt=0.1]


        return cls(*params, **params_dict)