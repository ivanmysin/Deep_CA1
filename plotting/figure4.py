import numpy as np
import matplotlib.pyplot as plt
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



T_st_idx = 1000 # start of t

hf = h5py.File('../outputs/firings/theta_freq_variation.h5', mode='r')

theta_freqs = sorted( hf.keys(),  key=lambda x: float(x) )
theta_phases4plots = np.linspace(-np.pi, np.pi, 100)
sine = 0.5 * (np.cos(theta_phases4plots) + 1)

fig, axes = plt.subplots(nrows=len(plotting_colors["neurons_order"]), ncols=len(theta_freqs)+1, \
                        constrained_layout=True, figsize=(15, 10)  )

for freq_idx, freq in enumerate( theta_freqs ):
    full_firings = hf[freq]['firings'][:]
    t = np.linspace(0, 1600, full_firings.shape[0])[T_st_idx:]

    freq = float(freq)

    theta_phases = (2 * np.pi * 0.001 * t * freq) % (2 * np.pi)
    theta_phases[theta_phases > np.pi] -= 2 * np.pi

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


        firings = full_firings[T_st_idx:, neuron_idx]
        firings_hist, firings_bins = np.histogram(theta_phases, bins=20, weights=firings, density=True, range=[-np.pi, np.pi])
        firings_bins = 0.5*(firings_bins[:-1] + firings_bins[1:])
        ax.plot(firings_bins, firings_hist, color=plotting_colors["neuron_colors"][neuron_name], linewidth=2, label="CBRD")


        sine_ampls = sine * 0.7 * np.max(firings_hist)
        ax.plot(theta_phases4plots, sine_ampls, linestyle="--", label="cos", color='black')
        ax.set_ylim(0, 0.35)
        ax.set_xlim(-np.pi, np.pi)




for ax2, neuron_name in zip(axes[:, 0], plotting_colors["neurons_order"]):
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(0.0, 0.5, neuron_name, fontsize=TEXTFONTSIZE)

hf.close()
fig.savefig(f'../outputs/plots/{fig_name}.png', dpi=500)
plt.show()