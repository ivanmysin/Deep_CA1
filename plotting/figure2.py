import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plots_config import plotting_colors
from sklearn.metrics import r2_score, d2_absolute_error_score
from scipy.ndimage import gaussian_filter1d

from myutils import get_phase_shift

import sys
sys.path.append('../')
from myutils import parzen_filter

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

neurons_order = plotting_colors["neurons_order"]

neurons_params = pd.read_excel('../parameters/neurons_parameters.xlsx', sheet_name='verified_theta_model')
neurons_params['Hippocampome_Neurons_Names'] = neurons_params['Hippocampome_Neurons_Names'].str.strip()
neurons_params['Model_Neurons_Names'] = neurons_params['Model_Neurons_Names'].str.strip()
# neurons_params['Simulated_Type'] = neurons_params['Simulated_Type'].str.strip()
neurons_params = neurons_params[neurons_params['Npops'] == 1]['Model_Neurons_Names'].to_list()
neuron_idx_in_sols = []
for neuron_name in neurons_order:
    neuron_idx_in_sols.append( neurons_params.index(neuron_name)  )

idxs_from_types = {
    'Basket': 1,
    'O-LM': 6,
    'Basket CCK+': 2,
    'Ivy': 4,
    'Neurogliaform': 5,
    'Bistratified': 3,
    'Axo-Axonic': 0,
    'Perforant Path-Associated':7,
    'Interneuron Specific R-O': 8,
    'Interneuron Specific RO-O':9,
}


dt = 0.01
duration = 2500

fig_name = 'fig2'

path_sim = '../outputs/firings/pop_theta_freq_variation.h5'
path_sim_units = '../outputs/results_freq8.h5'
path_dset = '../outputs/firings/dataset.h5'

hf = h5py.File(path_sim, 'r')
hf_units = h5py.File(path_sim_units, 'r')

hdf = h5py.File(path_dset, 'r')


gridspec_kw = {
    "width_ratios" : [1.0, 0.9, 1.0, 0.9],
}

if len(neurons_order)%2 == 0:
    nrows = len(neurons_order)//2
else:
    nrows = len(neurons_order) // 2 + 1

fig, axes = plt.subplots( nrows=nrows, ncols=4, \
                          gridspec_kw=gridspec_kw, constrained_layout=True, figsize=(18, 10))

#fig.tight_layout(pad=4.0)


full_firings = hf['8']['firings'][:]
full_targets = hdf['Ytrain'][:]


full_firings_units = hf_units['rate'][:]   #['8']['firings'][:]



# full_firings_units = full_firings_units.reshape(-1, len(neurons_order))

full_firings_units = parzen_filter(full_firings_units, window_size=505, axis=0)

full_targets = full_targets.reshape(-1, len(neurons_order))
full_targets = full_targets[: int(duration/dt), :]

t = np.linspace(0, full_targets.shape[0]*dt, full_targets.shape[0])
sine = 0.5 * (np.cos(2 * np.pi * 0.001*t * 8.0) + 1)

dt_dim = 0.01
t_units = np.linspace(0, full_firings_units.shape[0]*dt_dim, full_firings_units.shape[0])

for neuron_idx, neuron_name in enumerate(neurons_order):
     col_idx = 0
     row_idx = neuron_idx

     if neuron_idx > (nrows - 1):
         row_idx = neuron_idx - nrows

         col_idx = 2

     neuron_name_title = neuron_name
     if len(neuron_name_title) > 12:
         neuron_name_title = neuron_name_title[:5] + neuron_name_title[5:].replace(" ", "\n", 1)

     ax2 = axes[row_idx, col_idx]
     ax2.axis("off")
     ax2.set_xlim(0, 1)
     ax2.set_ylim(0, 1)
     ax2.text(0.5, 0.5, neuron_name_title, fontsize=TEXTFONTSIZE)

     ax = axes[row_idx, col_idx + 1]
     if neuron_idx == 0:
         ax.set_title("Частота разрядов")
     if (neuron_idx == nrows - 1) or (neuron_idx == len(neurons_order) - 1):
        ax.set_xlabel("Время (мс)")
     else:
        ax.xaxis.set_ticklabels([])

     if row_idx == int(nrows//2) :
         ax.set_ylabel("имп./сек.")

     # print(neuron_name)



     target = full_targets[:, neuron_idx_in_sols[neuron_idx]]

     firings = full_firings[:, neuron_idx_in_sols[neuron_idx]]



     units_idx = int( idxs_from_types[neuron_name] )
     # print(neuron_name, idxs_from_types[neuron_name])
     firings_units = full_firings_units[:,units_idx]

     # firings_units_smooth = gaussian_filter1d(firings_units, sigma=220)


     ax.plot(t, target, label = "Целевая частота", color='black', linewidth=5)
     ax.plot(t, firings, color=plotting_colors["neuron_colors"][neuron_name], linewidth=5, label="Симуляция")


     sine_ampls = sine * 0.7*np.max(firings)
     ax.plot(t, sine_ampls, linestyle="--", label = "cos", color='black')



     ax.plot(t_units, firings_units, color=plotting_colors["neuron_colors"][neuron_name], linewidth=2, linestyle="--", label="Точечные нейроны")


     f_max = 1.1* max( [np.max(firings[8000:]), np.max(target), np.quantile(firings_units[8000:], 0.85)])

     if f_max > 2 * np.max(target):
         f_max = 2 * np.max(target)

     ax.set_ylim(0, f_max)



     ax.legend( bbox_to_anchor=(1.7, 1.1), loc="upper right")
     ax.set_xlim(800, 1200)


     r2 = r2_score(target, firings)
     d2 = d2_absolute_error_score(target, firings)
     dPhi = get_phase_shift(target, firings, deg=True)

     print(f"{neuron_name} &  {r2:.2f} & {d2:.2f} & {dPhi:.0f}  " + r"\\\hline")

if len(neurons_order)%2 != 0:
    axes[-1, -1].axis('off')
    axes[-1, -2].axis('off')



fig.savefig(f'../outputs/plots/{fig_name}.png', dpi=200)
hf.close()

plt.show()

