import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import myconfig
from synapses_layers import TsodycsMarkramSynapse
from genloss import SpatialThetaGenerators
import os




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
            gen_model = SpatialThetaGenerators(gen_params)
            gen_model.precomute()
            self.generators.append(gen_model)


    # def get_generator_model(self, params):
    #     print(params)
    #
    #     #model = SpatialThetaGenerators(params)
    #
    #     return None


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

        #print(input_shape)

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

        model = tf.keras.Sequential()
        model.add(synapses_layer)

        for layer in base_model.layers:
            model.add(tf.keras.models.clone_model(layer))

        model.build(input_shape=input_shape)

        for l_idx, layer in enumerate(model.layers):
            if l_idx == 0:
                continue
            layer.trainable = False
            layer.set_weights(base_model.layers[l_idx - 1].get_weights())

        # print(model.summary())

        return tf.keras.models.clone_model(model)


    def call(self, firings0, t0=0, Nsteps=1):

        t = t0
        firings = []
        for idx in range(Nsteps):
            firings_in_step = []

            for model in self.pop_models:
                if idx == 0:
                    fired = model.predict(firings0)
                else:


                    fired = model.predict(firings[-1])
                firings_in_step.append(fired)

            for gen in self.generators:
                fired = gen.get_firings(t)
                #fired = self.ReshapeLayer(fired)

                fired = tf.reshape(fired, (1, 1, -1) )
                firings_in_step.append(fired)


            firings_in_step = self.ConcatLayer(firings_in_step)
            firings_in_step = self.ReshapeLayer( firings_in_step )
            firings.append(firings_in_step)

            t += self.dt

        firings = self.ConcatLayerInTime(firings)

        return firings




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

    firings0 = np.zeros(shape=(1, 1, len(populations)), dtype=np.float32)


    t = 0
    #fired = net.generators[0].get_firings(t)

    fired = net(firings0, t0=0, Nsteps=10)
    #
    # for m_idx, model in enumerate(net.pop_models):
    #     try:
    #         fired = model.predict(firings0)
    #
    #     except ValueError:
    #         print(m_idx, populations[m_idx]["type"])

    #fired = net.pop_models[6].predict(firings0)

    #

    #Y = net.pop_models[0].predict(firings0)

    print(fired.shape)