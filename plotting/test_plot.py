import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append('../')
from myutils import get_net_params

from plots_config import plotting_colors
import h5py

params = {'legend.fontsize': '16',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          }
plt.rcParams.update(params)
TEXTFONTSIZE = 'xx-large'

fig_name = 'fig4'


neuron_idx_in_sols = []
neurons_params = pd.read_excel('../parameters/neurons_parameters.xlsx', sheet_name='verified_theta_model')
neurons_params['Hippocampome_Neurons_Names'] = neurons_params['Hippocampome_Neurons_Names'].str.strip()
neurons_params['Model_Neurons_Names'] = neurons_params['Model_Neurons_Names'].str.strip()
neurons_params['Simulated_Type'] = neurons_params['Simulated_Type'].str.strip()
neurons_params = neurons_params[neurons_params['Npops'] == 1]['Model_Neurons_Names'].to_list()
for neuron_name in plotting_colors["neurons_order"]:
    neuron_idx_in_sols.append( neurons_params.index(neuron_name)  )

model_path =  '/home/ivan/nice_theta_models/5Hz_theta_model.keras'
params = get_net_params(model_path)

T_st_idx = 0 # start of t
theta_freq = 12
DT = 0.01
source_hfile = h5py.File('../outputs/firings/theta_freq_variation.h5', mode='r')

hf = source_hfile[f'{theta_freq}']

full_firings = hf['firings'][:, :]

print(params.keys())
gsyns = hf['A'][:] * params['gsyn_max']
e_r = params['e_r']

is_exc = (e_r > 0).astype(np.float32)
is_inh = (e_r < 0).astype(np.float32)

gsyn_exc = np.sum(gsyns * is_exc, axis=1)
gsyn_inh = np.sum(gsyns * is_inh, axis=1)

t = np.linspace(0, full_firings.shape[0]*DT, full_firings.shape[0])





sine = 0.5 * ( np.cos(2*np.pi*theta_freq*t*0.001) + 1)
#
#
# fig, axes = plt.subplots(nrows=len(source_hfile.keys()), ncols=1, \
#                                  constrained_layout=True, figsize=(5, 10), sharex=True  )
#
# for idx, theta_freq in enumerate( sorted( source_hfile.keys())):
#
#
#     generators_firings = source_hfile[f'{theta_freq}/generators_firings'][0, :, :]
#     axes[idx].set_title(theta_freq)
#     theta_freq = float(theta_freq)
#
#     if idx == 0:
#         t = np.linspace(0, generators_firings.shape[0]*DT, generators_firings.shape[0])
#         sine = 0.5 * ( np.cos(2*np.pi*theta_freq*t*0.001) + 1)
#
#
#     sine = 1.5* 0.5 * ( np.cos(2*np.pi*theta_freq*t*0.001) + 1)
#
#
#
#
#     axes[idx].plot(t, generators_firings[:, 0])
#     axes[idx].plot(t, generators_firings[:, 1])
#     axes[idx].plot(t, sine, color='black', linewidth=5)
#
#     axes[idx].set_xlim(0, 500)




#     axes[0].set_title(neuron_name)
#
#     ax = axes[0]
#
#     firings = full_firings[T_st_idx:, neuron_idx_in_sols[neuron_idx]]
#     cos_ref = sine * 0.7 * np.max(firings)
#
#
#     ax.plot(t, firings, color=plotting_colors["neuron_colors"][neuron_name], linewidth=2, label="Симуляция")
#     ax.plot(t, cos_ref, color='black', linewidth=0.5, label="Cos", linestyle='--')


for neuron_idx, neuron_name in enumerate(plotting_colors["neurons_order"]):

    fig, axes = plt.subplots(nrows=3, ncols=1, \
                             constrained_layout=True, figsize=(5, 10), sharex=True  )

    axes[0].set_title(neuron_name)

    ax = axes[0]

    firings = full_firings[T_st_idx:, neuron_idx_in_sols[neuron_idx]]
    cos_ref = sine * 0.7 * np.max(firings)


    ax.plot(t, firings, color=plotting_colors["neuron_colors"][neuron_name], linewidth=2, label="Симуляция")
    ax.plot(t, cos_ref, color='black', linewidth=0.5, label="Cos", linestyle='--')

    axes[1].plot(t, gsyn_exc[:, neuron_idx_in_sols[neuron_idx]], color='black', linewidth=1.5, label="Cos", linestyle='--')
    axes[2].plot(t, gsyn_inh[:, neuron_idx_in_sols[neuron_idx]], color='black', linewidth=1.5, label="Cos", linestyle='--')


    for pre_idx, pre_name in enumerate(neurons_params):

        if e_r[pre_idx, 0] > 0:
            ax = axes[1]
        else:
            ax = axes[2]

        gsyn = gsyns[:, pre_idx,  neuron_idx_in_sols[neuron_idx]]

        #gsyn_exc = gsyns[:, -2:, neuron_idx]

        ax.plot(t, gsyn, label=pre_name)


    ax.set_xlim(0, 1500)

source_hfile.close()
#fig.savefig(f'../outputs/plots/{fig_name}.png', dpi=500)
plt.show()