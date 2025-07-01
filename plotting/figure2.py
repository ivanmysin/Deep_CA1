import numpy as np
import matplotlib.pyplot as plt
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



dt = 0.01
duration = 2400

fig_name = 'fig2'


neurons_order = plotting_colors["neurons_order"]
path = '../outputs/firings/2_output.h5'

hf = h5py.File(path, 'r')
t = np.linspace(0, duration, 240000 )
sine = 0.5 * (np.cos(2 * np.pi * 0.001*t * 8.0) + 1)

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




for neuron_idx, neuron_name in enumerate(neurons_order):
     col_idx = 0
     row_idx = neuron_idx

     if neuron_idx > (nrows - 1):
         row_idx = neuron_idx - nrows

         col_idx = 2

     neuron_name_title = neuron_name
     if len(neuron_name_title) > 17:
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

     print(neuron_name)

     firings = hf[neuron_name]['firings'][:]
     target = hf[neuron_name]['target_firing'][:]

     # print("target", target.shape)
     # print("firings", firings.shape)
     ax.plot(t, target, label = "Целевая частота", color='black', linewidth=4)
     ax.plot(t, firings, color=plotting_colors["neuron_colors"][neuron_name], linewidth=5, label="Симуляция")
#     neurons_indexes = montecarlofile[neuron_name + "_indexes"][:]
#     neurons_times = montecarlofile[neuron_name + "_times"][:]
#     montecarlofirings, _ = np.histogram(neurons_times, bins=t)
#     montecarlofirings = montecarlofirings / np.max(neurons_indexes + 1)
#     montecarlofirings = montecarlofirings / (0.001 * (t[1] - t[0]))
#     montecarlofirings = np.convolve(montecarlofirings, Parzen, mode='same')
#
#     #ax.plot(t[:-1], montecarlofirings, linestyle="--", color=plotting_colors["neuron_colors"][neuron_name], label="Monte-Carlo")
     sine_ampls = sine * 0.7*np.max(target)
     ax.plot(t, sine_ampls, linestyle="--", label = "cos", color='black')

     ax.set_ylim(0, 1.1*max( [np.max(firings[8000:]), np.max(target)]) )



     ax.legend( bbox_to_anchor=(1.7, 1.1), loc="upper right")
     ax.set_xlim(800, 1200)

if len(neurons_order)%2 != 0:
    axes[-1, -1].axis('off')
    axes[-1, -2].axis('off')



fig.savefig(f'../outputs/plots/{fig_name}.png', dpi=200)
hf.close()

plt.show()