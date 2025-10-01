import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

T_st_idx = 200000 # start of t

source_hfile = h5py.File('../outputs/firings/theta_freq_variation.h5', mode='r')

theta_freqs = sorted( source_hfile.keys(),  key=lambda x: float(x) )
theta_phases4plots = np.linspace(-np.pi, np.pi, 100)
sine = 0.5 * (np.cos(theta_phases4plots) + 1)

duration = 2500
DT = 0.01

fig, axes = plt.subplots(nrows=len(plotting_colors["neurons_order"]), ncols=len(theta_freqs)+1, \
                        constrained_layout=True, figsize=(15, 10)  )

for freq_idx, freq in enumerate( theta_freqs ):
    print(freq)
    full_firings = source_hfile[freq]['firings'][:]

    t = np.linspace(0, duration, full_firings.shape[0])[T_st_idx:]

    freq = float(freq)
    theta_phases = (2 * np.pi * 0.001 * t * freq) % (2 * np.pi)
    theta_phases[theta_phases > np.pi] -= 2 * np.pi

    # firings = full_firings
    #
    # cos_ref = np.max(firings) * 0.5 * (np.cos(theta_phases) + 1)
    #
    # fig2, axes2 = plt.subplots(nrows=2)
    # axes2[0].plot(t, firings)
    # axes2[0].plot(t, cos_ref)
    #
    # #axes2[1].plot(firings_bins, firings_hist)
    # plt.show()
    #
    # continue

    for neuron_idx, neuron_name in enumerate(plotting_colors["neurons_order"]):

        ax = axes[neuron_idx, freq_idx+1]

        if neuron_idx == 0:
            ax.set_title(str(int(freq)) + "Гц")

        if (neuron_idx == len(plotting_colors["neurons_order"]) - 1) and (freq_idx == int(len(theta_freqs) // 2)):
            ax.set_xlabel("Фаза (рад)")

        if neuron_idx < (len(plotting_colors["neurons_order"]) - 1):
            ax.xaxis.set_ticklabels([])

        if (neuron_idx == int(len(plotting_colors["neurons_order"]) // 2)) and (freq_idx == 0):
            ax.set_ylabel("Гц/рад")

        if freq_idx > 0:
            ax.yaxis.set_ticklabels([])


        firings = full_firings[T_st_idx:, neuron_idx_in_sols[neuron_idx]]

        firings_hist, firings_bins = np.histogram(theta_phases, bins=20, weights=firings, density=True, range=[-np.pi, np.pi])
        firings_bins = 0.5*(firings_bins[:-1] + firings_bins[1:])
        ax.plot(firings_bins, firings_hist, color=plotting_colors["neuron_colors"][neuron_name], linewidth=2, label="Симуляция")



        sine_ampls = sine * 0.7 * np.max(firings_hist)
        ax.plot(theta_phases4plots, sine_ampls, linestyle="--", label="cos", color='black')
        ax.set_ylim(0, 0.35)
        ax.set_xlim(-np.pi, np.pi)




for ax2, neuron_name in zip(axes[:, 0], plotting_colors["neurons_order"]):
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(0.0, 0.5, neuron_name, fontsize=TEXTFONTSIZE)

source_hfile.close()
fig.savefig(f'../outputs/plots/{fig_name}.png', dpi=500)
plt.show()