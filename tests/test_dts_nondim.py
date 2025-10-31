import pandas as pd
import os

os.chdir('../')
import myconfig

neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
neurons_params.rename(
    {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
     'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

populations = pd.read_excel(myconfig.FIRINGSNEURONPARAMS, sheet_name='verified_theta_model')
populations.rename({'neurons': 'type'}, axis=1, inplace=True)
populations = populations[populations['Npops'] > 0]
populations = populations[populations['Simulated_Type'] == 'simulated']


for idx, pop in populations.iterrows():

    pop_name = pop['Hippocampome_Neurons_Names']

    p = neurons_params[neurons_params["Neuron Type"] == pop_name]

    Cm = p['Cm'].values
    k = p['k'].values
    Vrest = p['Vrest'].values

    koeff = k * abs(Vrest) / Cm

    Delta_eta = 80 /  (2 * koeff)

    print(pop_name)
    print('Delta_eta =', Delta_eta[0])
    print('koeff =', koeff[0])
    print('Cm =', Cm[0])
    print('Vrest =', Vrest[0])
    print('k =', k[0])
    print('='*20)