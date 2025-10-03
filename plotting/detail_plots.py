import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plots_config import plotting_colors

import sys
sys.path.append('../')
import myutils

import h5py
params = {'legend.fontsize': '12',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large',
          }
plt.rcParams.update(params)
TEXTFONTSIZE = 'xx-large'

model_filepath = '../outputs/big_models/theta_model_4090.keras'
params = myutils.get_net_params(model_filepath)

neurons_params = pd.read_excel('../parameters/neurons_parameters.xlsx', sheet_name='verified_theta_model')
neurons_params['Hippocampome_Neurons_Names'] = neurons_params['Hippocampome_Neurons_Names'].str.strip()
neurons_params['Model_Neurons_Names'] = neurons_params['Model_Neurons_Names'].str.strip()
neurons_params['Simulated_Type'] = neurons_params['Simulated_Type'].str.strip()
neurons_params = neurons_params[neurons_params['Npops'] == 1]['Model_Neurons_Names'].to_list()

# neurons_names = neurons_params[neurons_params['Npops'] == 1]['Hippocampome_Neurons_Names'].to_list()
#
#
# neurons_params_values = pd.read_csv('../parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv')
# neurons_params_values.rename(
#     {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
#      'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)
#
# for idx, neuron_name in enumerate(neurons_names):
#
#     nv = neurons_params_values[neurons_params_values['Neuron Type'] == neuron_name]
#
#     delta_eta = params['Delta_eta'][idx] * nv['k'].values[0] * (nv['Vrest'].values[0]**2)
#
#     print(neuron_name, delta_eta)



neuron_idx_in_sols = []
for neuron_name in plotting_colors["neurons_order"]:
    neuron_idx_in_sols.append( neurons_params.index(neuron_name)  )


dt = 0.01
fig_name = 'detail_plots'

neurons_order = plotting_colors["neurons_order"]
path = '../outputs/firings/theta_freq_variation.h5'

theta_freq = 4.0

hf = h5py.File(path, 'r')



firings = hf[str(int(theta_freq))]['firings'][:]
v_avg = hf[str(int(theta_freq))]['v_avg'][:]
w_avg = hf[str(int(theta_freq))]['w_avg'][:]

A = hf[str(int(theta_freq))]['A'][:]
gsyn = A * params['gsyn_max']

#print(A.shape, params['gsyn_max'].shape)
gsyn_tot = np.sum(gsyn, axis=1)
is_exc = (params['e_r'] > 0).astype(np.float32)
is_inh = (params['e_r'] < 0).astype(np.float32)

gsyn_exc = np.sum(gsyn * is_exc, axis=1)
gsyn_inh = np.sum(gsyn * is_inh, axis=1)

Isyn = np.sum(gsyn * (params['e_r'] - v_avg), axis=1)

t = np.linspace(0, firings.shape[0]*dt, firings.shape[0])
sine = 0.5 * (np.cos(2 * np.pi * 0.001*t * theta_freq) + 1)


hf.close()

for neuron_idx, neuron_name in enumerate(neurons_order):


    fig, axes = plt.subplots( nrows=4, ncols=1, \
                              constrained_layout=True, figsize=(18, 10))

    fig.suptitle(f'{neuron_name}', fontsize=TEXTFONTSIZE)

    firings_pops = firings[:, neuron_idx_in_sols[neuron_idx]]
    sine_pops = sine * 0.7*np.max(firings_pops)

    v_avg_pops = v_avg[:, 0, neuron_idx_in_sols[neuron_idx]]
    w_avg_pops = -w_avg[:, 0, neuron_idx_in_sols[neuron_idx]]

    # target = hf[neuron_name]['target_firing'][:]

    # print("target", target.shape)
    # print("firings", firings.shape)
    #ax.plot(t, target, label = "Целевая частота", color='black', linewidth=4)
    axes[0].plot(t, firings_pops, color=plotting_colors["neuron_colors"][neuron_name], linewidth=5, label="Симуляция")
    axes[0].plot(t, sine_pops, color='k', linewidth=1, label="cos", linestyle="--")

    axes[1].plot(t, v_avg_pops, color=plotting_colors["neuron_colors"][neuron_name], linewidth=5, label="Симуляция")
    axes[2].plot(t, w_avg_pops, color=plotting_colors["neuron_colors"][neuron_name], linewidth=5, label="Симуляция")

    # axes[3].plot(t, gsyn_tot[:, neuron_idx_in_sols[neuron_idx]], color=plotting_colors["neuron_colors"][neuron_name], linewidth=5, label="Gsyn")


    isine = sine * 0.7 * np.max(gsyn_exc[:, neuron_idx_in_sols[neuron_idx]])
    axes[3].plot(t, gsyn_exc[:, neuron_idx_in_sols[neuron_idx]], color='red', linewidth=5, label="Exc")
    axes[3].plot(t, isine, color='k', linewidth=2, label="cos", linestyle="--")
    # axes[3].plot(t, gsyn_inh[:, neuron_idx_in_sols[neuron_idx]], color='blue', linewidth=5, label="Inh")

    # axes[3].plot(t, Isyn[:, neuron_idx_in_sols[neuron_idx]], color=plotting_colors["neuron_colors"][neuron_name], linewidth=5, label="Isyn")
    axes[3].legend(loc='upper right', fontsize=TEXTFONTSIZE)

    for ax in axes:
        ax.set_xlim(1000, 1500)

    plt.show()




    #fig.savefig(f'../outputs/plots/{fig_name}.png', dpi=200)



