import numpy as np
import matplotlib.pyplot as plt
from plots_config import plotting_colors

import h5py

params = {'legend.fontsize': '16',
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

fig_name = 'fig3.2'

neurons_order = plotting_colors["neurons_order"][8 : ]

path = '../outputs/firings/output.h5'
hf = h5py.File(path, 'r')

t = np.linspace(0, duration, 240000 )
sine = 0.5 * (np.cos(2 * np.pi * 0.001*t * 8.0) + 1)



gridspec_kw = {
    "width_ratios" : [0.15, ].extend([0.1 for _ in range(len(neurons_order))]),
}
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 15), gridspec_kw=gridspec_kw)


for neuron_idx, neuron_name in enumerate(neurons_order):



    if neuron_idx < 4:
        row_idx = 0
        plot_idx = neuron_idx + 1
    else:
        row_idx = 2
        plot_idx = neuron_idx + 1 - 4

    axes[row_idx, plot_idx].set_title(neuron_name)

    if neuron_idx == 0 or neuron_idx == 4:
        axes[row_idx, plot_idx].set_ylabel(r"$\mu S$")
        axes[row_idx+1, plot_idx].set_ylabel(r"$\mu S$")

    if row_idx == 2 or plot_idx == 4:
        axes[row_idx+1, plot_idx].set_xlabel("Time (ms)")
    else:
        axes[row_idx+1, plot_idx].xaxis.set_ticklabels([])



    for pre_name in hf[neuron_name].keys():
        if pre_name == 'CA3_generator':
            pre_name = 'CA3 Pyramidal'

        if pre_name == 'MEC_generator':
            pre_name = 'EC LIII Pyramidal'

        if not ((pre_name in plotting_colors["neurons_order"]) or (pre_name in plotting_colors["generators_order"])):
            continue


        if pre_name == "CA3 Pyramidal":
            g_syn = hf[neuron_name]['CA3_generator'][:]
        elif pre_name == "EC LIII Pyramidal":
            g_syn = hf[neuron_name]['MEC_generator'][:]
        else:
            g_syn =  hf[neuron_name][pre_name][:]
        if pre_name in ["CA1 Pyramidal (deep)", "CA1 Pyramidal (superficial)" , "CA3 Pyramidal", "EC LIII Pyramidal"] :
            ax = axes[row_idx, plot_idx]
        else:
            ax = axes[row_idx+1, plot_idx]


        color = plotting_colors["neuron_colors"][pre_name]
        ax.plot(t, g_syn, linestyle="-", label=pre_name, color=color)

    exc_g = hf[neuron_name]['gsyn_exc'][:]
    inh_g = hf[neuron_name]['gsyn_inh'][:]

    axes[row_idx, plot_idx].plot(t, exc_g, linestyle=(0, (1, 1)), label="sum exc", color='orange', linewidth=2)
    axes[row_idx+1, plot_idx].plot(t, inh_g, linestyle=(0, (1, 1)), label="sum inh", color='magenta', linewidth=2)

    sine_amples_exc = 0.7*np.max(exc_g[10000:]) * sine
    axes[row_idx, plot_idx].plot(t, sine_amples_exc, linestyle='--', label="cos", color='black')

    sine_amples_inh = 0.7*np.max(inh_g[10000:]) * sine
    axes[row_idx+1, plot_idx].plot(t, sine_amples_inh, linestyle="--", label="cos", color='black')

    axes[row_idx, plot_idx].set_ylim(0.0, 1.1*np.max(exc_g[10000:]))
    axes[row_idx+1, plot_idx].set_ylim(0.0, 1.1*np.max(inh_g[10000:]))

for ax1 in axes:
    for ax in ax1:
        ax.set_xlim(800, 1200)

lines = []
labels = ['sum exc', 'sum inh'] +  plotting_colors["neurons_order"] +  plotting_colors["generators_order"]


for label in labels:
    for ax in fig.axes:
        Line, Label = ax.get_legend_handles_labels()
        try:
            line_idx = Label.index(label)
            lines.append(Line[line_idx])
            #print(label)
            break
        except ValueError:
            continue

axes[2, -1].axis("off")
axes[3, -1].axis("off")


axes[3, -1].legend(lines, labels,  ncol=2, loc='lower left') # , bbox_to_anchor =(0.15, 0.1),


ax0 = axes[0, 0]
ax0.axis("off")
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.text(0.0, 0.5, " Conductivity of \n exciting inputs", fontsize=TEXTFONTSIZE)

ax0 = axes[1, 0]
ax0.axis("off")
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.text(0.0, 0.5, " Conductivity of \n inhibitory inputs", fontsize=TEXTFONTSIZE)

ax0 = axes[2, 0]
ax0.axis("off")
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.text(0.0, 0.5, " Conductivity of \n exciting inputs", fontsize=TEXTFONTSIZE)

ax0 = axes[3, 0]
ax0.axis("off")
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.text(0.0, 0.5, " Conductivity of \n inhibitory inputs", fontsize=TEXTFONTSIZE)

fig.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5, left=0.05, right=0.9)
#fig.tight_layout()

fig.savefig(f'../outputs/plots/{fig_name}.png', dpi=500)
hf.close()
plt.show()