import numpy as np
import tensorflow as tf
from keras import saving
import pandas as pd
import pickle

import myconfig
from synapses_layers import TsodycsMarkramSynapse
import genloss  #s import SpatialThetaGenerators
import os

tf.keras.backend.set_floatx('float32')

from pprint import pprint

class PopModelLayer(tf.keras.layers.Layer):

    def __init__(self, synapse_layer, pop_layers):
        super(PopModelLayer, self).__init__()
        self.synapses = synapse_layer
        self.pop_layers = pop_layers

    def build(self, input_shape):
        self.synapses.build(input_shape)
        self.built = True



    def call(self, firings):

        x = self.synapses(firings)
        for l in self.pop_layers:
            x = l(x)

        return x


    def get_config(self):
        config = super().get_config()

        pop_layers_config = []
        for l in self.pop_layers:
            pop_layers_config.append( saving.serialize_keras_object(l)  )

        config.update({
            "synapses_config": saving.serialize_keras_object(self.synapses),
            "pop_layers_config" : pop_layers_config,

        })

        ## pprint(config)
        return config

    @classmethod
    def from_config(cls, config):

        #pprint(config)

        synapses = saving.deserialize_keras_object( config.pop("synapses_config") )

        pop_layers = []
        for pop_config in config.pop("pop_layers_config"):
            l = saving.deserialize_keras_object(  pop_config )

            pop_layers.append(l)

        return cls(synapses, pop_layers)

class Net(tf.keras.Model):

    def __init__(self, populations, connections, pop_types_params, neurons_params, synapses_params):
        super(Net, self).__init__()

        self.dt = myconfig.DT
        self.Npops = len(populations)

        simulated_types = pop_types_params[pop_types_params["is_include"] == 1]["neurons"].to_list()

        pop_types_models = self.get_pop_types_models(myconfig.PRETRANEDMODELS, simulated_types)

        self.ConcatLayer = tf.keras.layers.Concatenate(axis=-1)
        self.ConcatLayerInTime = tf.keras.layers.Concatenate(axis=1)
        self.ReshapeLayer = tf.keras.layers.Reshape( (1, -1) )

        self.pop_models = []
        gen_params = []
        for pop_idx, pop in enumerate(populations):
            if "_generator" in pop["type"]:
               gen_params.append(pop)
            else:
                base_model = pop_types_models[pop["type"]]
                pop_model = self.get_model(pop_idx, pop, connections, base_model, neurons_params, synapses_params)
                self.pop_models.append(pop_model)


        self.generators = []
        if len(gen_params) > 0:
            gen_model = genloss.SpatialThetaGenerators(gen_params)
            gen_model.build()
            self.generators.append(gen_model)

        self.output_layers = self.get_output_layers(populations)

        self.firings_decorrelator = genloss.Decorrelator(strength=0.001) #strength можно определить в myconf



    def get_output_layers(self, populations):

        simple_out_mask = np.zeros(len(populations), dtype='bool')
        frequecy_filter_out_mask = np.zeros(len(populations), dtype='bool')
        phase_locking_out_mask = np.zeros(len(populations), dtype='bool')

        for pop_idx, pop in enumerate(populations):

            if pop["type"] == "CA1 Pyramidal":
                simple_out_mask[pop_idx] = True
                continue
            try:

                if np.isnan(pop["ThetaPhase"]):
                    phase_locking_out_mask[pop_idx] = True
                    continue
                else:
                    frequecy_filter_out_mask[pop_idx] = True
                    continue


            except KeyError:
                continue


        output_layers = []
        simple_selector = genloss.CommonOutProcessing(simple_out_mask)
        output_layers.append(simple_selector)


        if np.sum(frequecy_filter_out_mask) > 0:
            theta_filter = genloss.FrequencyFilter(mask=frequecy_filter_out_mask, dt=myconfig.DT)
            output_layers.append(theta_filter)

            robast_mean_out = genloss.RobastMeanOut(mask=frequecy_filter_out_mask)
            output_layers.append(robast_mean_out)


        if np.sum(phase_locking_out_mask) > 0:
            phase_locking_selector = genloss.PhaseLockingOutput(mask=phase_locking_out_mask,
                                                                ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT)
            output_layers.append(phase_locking_selector)


        ### add robast mean!!!!
        return output_layers


    def get_pop_types_models(self, path, types):
        pop_types_models = {}

        for pop_type in types:
            modelfile = path  + pop_type + ".keras"

            try:
                pop_types_models[pop_type] = tf.keras.models.load_model(modelfile)
            except ValueError:
                print(f"File for model population {pop_type} is not found")

        return pop_types_models


    def get_model(self, pop_idx, pop, connections, base_model, neurons_params, synapses_params):

        input_shape = (1, None, self.Npops)


        pop_type = pop["type"]



        if len( neurons_params[neurons_params["Neuron Type"] == pop_type] ) == 0:
            print(pop_type)

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

        is_connected_mask = np.zeros(self.Npops, dtype='bool')

        for conn in connections:
            if conn["post_idx"] != pop_idx: continue

            syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == conn['pre_type']) & (
                    synapses_params['Postsynaptic Neuron Type'] == conn['post_type'])]

            if len(syn) == 0:
                continue

            is_connected_mask[conn["pre_idx"]] = True

            conn_params["gsyn_max"].append(np.random.rand())  # !!!
            conn_params['pconn'].append(conn['pconn'])

            Uinc = syn['Uinc'].values[0]
            tau_r = syn['tau_r'].values[0]
            tau_f = syn['tau_f'].values[0]
            tau_d = syn['tau_d'].values[0]

            if neurons_params[neurons_params['Neuron Type'] == conn['pre_type']]['E/I'].values[0] == "e":
                Erev = 0
            elif neurons_params[neurons_params['Neuron Type'] == conn['pre_type']]['E/I'].values[0] == "i":
                Erev = -75.0

            conn_params['Uinc'].append(Uinc)
            conn_params['tau_r'].append(tau_r)
            conn_params['tau_f'].append(tau_f)
            conn_params['tau_d'].append(tau_d)
            conn_params['Erev'].append(Erev)


        if np.sum(is_connected_mask) == 0:
            print("Not connected", pop["type"])

        synapses = TsodycsMarkramSynapse(conn_params, dt=self.dt, mask=is_connected_mask)
        synapses_layer = tf.keras.layers.RNN(synapses, return_sequences=True, stateful=True)

        pop_layers = []
        for layer in base_model.layers:
            pop_layers.append( tf.keras.models.clone_model(layer) )

        model = PopModelLayer(synapses_layer, pop_layers)   # tf.keras.Sequential()

        model.build(input_shape=input_shape)

        for l_idx, layer in enumerate(model.pop_layers):
            layer.trainable = False
            layer.set_weights(base_model.layers[l_idx].get_weights())

        # print(model.summary())


        return tf.keras.models.clone_model(model)



    def simulate(self,  firings0, t0=0, Nsteps=1):
        t = tf.constant(t0, dtype=tf.float32)
        firings = []
        for idx in range(Nsteps):

            firings_in_step = []

            for model in self.pop_models:
                if idx == 0:
                    fired = model(firings0)
                else:

                    fired = model(firings[-1])
                firings_in_step.append(fired)

            for gen in self.generators:
                t = tf.reshape(t, shape=(-1, 1))
                fired = gen(t)
                # fired = self.ReshapeLayer(fired)

                fired = tf.reshape(fired, (1, 1, -1))
                firings_in_step.append(fired)

            firings_in_step = self.ConcatLayer(firings_in_step)
            firings_in_step = self.ReshapeLayer(firings_in_step)
            firings.append(firings_in_step)

            t += self.dt

        firings = self.ConcatLayerInTime(firings)
        return firings


    def call(self, firings0, t0=0, Nsteps=1):


        firings = self.simulate(firings0, t0=t0, Nsteps=Nsteps)
        corr_penalty = self.firings_decorrelator( firings )
        self.add_loss(corr_penalty)


        outputs = []
        for out_layer in self.output_layers:
            out = out_layer(firings)
            outputs.append(out)

        return outputs




if __name__ == "__main__":
    with open(myconfig.STRUCTURESOFNET + "_neurons.pickle", "rb") as neurons_file:
        populations = pickle.load(neurons_file)


    types = set( [pop['type'] for pop in populations] )
    #print(types)

    with open(myconfig.STRUCTURESOFNET + "_connections.pickle", "rb") as synapses_file:
        connections = pickle.load(synapses_file)



    pop_types_params = pd.read_excel(myconfig.SCRIPTS4PARAMSGENERATION + "neurons_parameters.xlsx", sheet_name="Sheet2",
                               header=0)

    neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
    neurons_params.rename({'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd', 'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)
    synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
    synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)

    for pop_idx, pop in enumerate(populations):
        pop_type = pop["type"]
        is_connected_mask = np.zeros(len(populations), dtype='bool')

        for conn in connections:
            if conn["post_idx"] != pop_idx: continue
            is_connected_mask[conn["pre_idx"]] = True


        if np.sum(is_connected_mask) == 0:
            conn = {
                "pconn" : 0.0,
                "pre_idx" : 0,
                "post_idx" : pop_idx,
                "pre_type" : populations[0],
                "post_type" : populations[pop_idx],
            }
            connections.append(conn)


    print("############################################")
    net = Net(populations, connections, pop_types_params, neurons_params, synapses_params)

    test_input = np.random.uniform(0.5, 1, size=len(populations)).reshape(1, 1, -1)
    test_input = tf.constant(test_input)


    # test_out = net.pop_models[10](test_input)
    # print(test_out)

    t = tf.constant(0.0, dtype=tf.float32)
    t = tf.reshape(t, shape=(-1, 1))

    # outs = net.generators[0](t)
    #
    # print(outs)

    firings0 = np.zeros(shape=(1, 1, len(populations)), dtype=np.float32)
    #firings0 = np.random.uniform(0, 0.01, len(populations)).astype(np.float32).reshape(1, 1, -1)


    t = 0.0
    #fired = net.generators[0].get_firings(t)

    fired = net(firings0, t0=0.0, Nsteps=10)
    print(fired[0])