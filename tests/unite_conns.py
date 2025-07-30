import pandas as pd
import os

os.chdir('../')
import myconfig

synapses_params = pd.read_csv(myconfig.TSODYCSMARKRAMPARAMS)
synapses_params.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)

keys = ["gsyn_max", "Uinc", "tau_d", "tau_r", "tau_f"]

for key in keys:
    kmean = synapses_params[key].mean()
    kmedian = synapses_params[key].median()
    kmax = synapses_params[key].max()
    kmin = synapses_params[key].min()

    print(key, kmean, kmedian, kmin, kmax)


potconn_file_path = './parameters/PetentialConnections.csv'
potential_connections = pd.read_csv(potconn_file_path)
potential_connections['Presynaptic Neuron Type'] = potential_connections['Presynaptic Neuron Type'].str.strip()
potential_connections['Postsynaptic Neuron Type'] = potential_connections['Postsynaptic Neuron Type'].str.strip()



populations = pd.read_excel(myconfig.FIRINGSNEURONPARAMS, sheet_name='verified_theta_model')
populations = populations[populations['Npops'] > 0]

for pre_idx, (_, pre_pop) in enumerate(populations.iterrows()):
    for post_idx, (_, post_pop) in enumerate(populations.iterrows()):
        if post_pop['Simulated_Type'] == 'generator':
            continue

        pre_type = pre_pop['Hippocampome_Neurons_Names']
        post_type = post_pop['Hippocampome_Neurons_Names']

        syn = synapses_params[(synapses_params['Presynaptic Neuron Type'] == pre_type) & (
                synapses_params['Postsynaptic Neuron Type'] == post_type)]

        if len(syn) == 0:
            syn = potential_connections[(potential_connections['Presynaptic Neuron Type'] == pre_type) & (
                    potential_connections['Postsynaptic Neuron Type'] == post_type)]

            if len(syn) == 0:
                continue
            else:
                print(pre_type, '->', post_type, syn)


