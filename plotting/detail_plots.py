import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plots_config import plotting_colors

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

# neuron_idx_in_sols = []
neurons_params = pd.read_excel('../parameters/neurons_parameters.xlsx', sheet_name='verified_theta_model')
neurons_params['Hippocampome_Neurons_Names'] = neurons_params['Hippocampome_Neurons_Names'].str.strip()
neurons_params['Model_Neurons_Names'] = neurons_params['Model_Neurons_Names'].str.strip()
neurons_params['Simulated_Type'] = neurons_params['Simulated_Type'].str.strip()
neurons_params = neurons_params[neurons_params['Npops'] == 1]['Model_Neurons_Names'].to_list()

neuron_idx_in_sols = []
for neuron_name in plotting_colors["neurons_order"]:
    neuron_idx_in_sols.append( neurons_params.index(neuron_name)  )


dt = 0.01
duration = 2500

fig_name = 'detail_plots'

neurons_order = plotting_colors["neurons_order"]
path = '../outputs/firings/theta_freq_variation.h5'

theta_freq = 12.0

hf = h5py.File(path, 'r')



firings = hf[str(int(theta_freq))]['firings'][:]
v_avg = hf[str(int(theta_freq))]['v_avg'][:]
w_avg = hf[str(int(theta_freq))]['w_avg'][:]

t = np.linspace(0, firings.shape[0]*dt, firings.shape[0])
sine = 0.5 * (np.cos(2 * np.pi * 0.001*t * theta_freq) + 1)


for neuron_idx, neuron_name in enumerate(neurons_order):


    # gridspec_kw = {
    #     "width_ratios" : [1.0, 0.9, 1.0, 0.9],
    # }
    #
    # if len(neurons_order)%2 == 0:
    #     nrows = len(neurons_order)//2
    # else:
    #     nrows = len(neurons_order) // 2 + 1

    fig, axes = plt.subplots( nrows=4, ncols=1, \
                              constrained_layout=True, figsize=(18, 10))  #

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


    for ax in axes:
        ax.set_xlim(1000, 1500)

    plt.show()




#fig.savefig(f'../outputs/plots/{fig_name}.png', dpi=200)
hf.close()



