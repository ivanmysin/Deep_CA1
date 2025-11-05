import numpy as np
import pandas as pd
import os

os.chdir('../')
import myconfig

neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
neurons_params.rename(
    {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
     'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

populations = pd.read_excel(myconfig.FIRINGSNEURONPARAMS, sheet_name='verified_theta_model')
populations.rename({'neurons': 'type'}, axis=1, inplace=True)
populations = populations[populations['Npops'] > 0]
populations = populations[populations['Simulated_Type'] == 'simulated']


for idx, pop in populations.iterrows():

    pop_name = pop['Hippocampome_Neurons_Names']

    p = neurons_params[neurons_params["Neuron Type"] == pop_name]

    Cm = p['Cm'].values[0]
    k = p['k'].values[0]
    Vrest = p['Vrest'].values[0]
    Vth = p['Vth'].values[0]

    dt_non_dim_koeff = k * abs(Vrest) / Cm

    alpha = 1 + Vth/(np.abs(Vrest))

    mean_target_fr = pop['OutPlaceFiringRate']

    rst = mean_target_fr / 1000 / dt_non_dim_koeff

    # Delta_eta = Cm / 4  # 80 /  (2 * dt_non_dim_koeff) # pop['Delta_eta']  #  mean_target_fr * np.pi * 0.001 * alpha      #

    Delta_eta = pop['Delta_eta']

    Delta_eta = Delta_eta / (k * Vrest**2)

    vst = 0.5*(alpha - Delta_eta/np.pi/rst)

    print(pop_name)
    print('Delta_eta =', Delta_eta)
    print('koeff =', dt_non_dim_koeff)
    print('Cm =', Cm)
    print('Vrest =', Vrest)
    print('k =', k)
    print('VT =', Vth)
    print('alpha =', alpha )
    print('vst =', vst )
    print('='*20)