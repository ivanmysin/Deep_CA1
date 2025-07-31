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
duration = 2500

fig_name = 'fig3'

neurons_order = plotting_colors["neurons_order"]

path = '../outputs/firings/output.h5'
hf = h5py.File(path, 'r')

t = np.linspace(0, duration, int(duration / dt) )
sine = 0.5 * (np.cos(2 * np.pi * 0.001*t * 8.0) + 1)



gridspec_kw = {
    "width_ratios" : [0.15, ] + [0.1, ] * 4 + [0.1, ]
}

nrows = 6
fig, axes = plt.subplots(nrows=nrows, ncols=6, figsize=(20, 15), gridspec_kw=gridspec_kw)


for neuron_idx, neuron_name in enumerate(neurons_order):

    plot_idx = neuron_idx%4 + 1

    row_idx = 2 * int(neuron_idx / 4)

    if neuron_idx == len(neurons_order) - 1:
        row_idx = 0
        plot_idx = 5


    neuron_name_title = neuron_name
    if len(neuron_name_title) > 12:
        neuron_name_title = neuron_name_title[:5] +  neuron_name_title[5:].replace(" ", "\n", 1)


    axes[row_idx, plot_idx].set_title(neuron_name_title)

    # if neuron_idx == 0 or neuron_idx == 4:
    #     axes[row_idx, plot_idx].set_ylabel(r"")
    #     axes[row_idx+1, plot_idx].set_ylabel(r"")

    if row_idx == (nrows - 2) or neuron_idx == len(neurons_order) - 1:
        axes[row_idx+1, plot_idx].set_xlabel("Время (мс)")
    else:
        axes[row_idx+1, plot_idx].xaxis.set_ticklabels([])

    axes[row_idx, plot_idx].xaxis.set_ticklabels([])



    for pre_name in hf[neuron_name].keys():
        if pre_name == 'CA3 Input':
            pre_name = 'CA3 Pyramidal'

        if pre_name == 'MEC Input':
            pre_name = 'EC LIII Pyramidal'

        if not ((pre_name in plotting_colors["neurons_order"]) or (pre_name in plotting_colors["generators_order"])):
            continue


        if pre_name == "CA3 Pyramidal":
            g_syn = hf[neuron_name]['CA3 Input'][:]
        elif pre_name == "EC LIII Pyramidal":
            g_syn = hf[neuron_name]['MEC Input'][:]
        else:
            g_syn =  hf[neuron_name][pre_name][:]
        if pre_name in ["Pyramidal (deep)", "Pyramidal (superficial)" , "CA3 Pyramidal", "EC LIII Pyramidal"] :
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

    max_exc_g = np.max(exc_g[10000:])
    if max_exc_g > 0.00001:
        axes[row_idx, plot_idx].set_ylim(0.0, 1.1*max_exc_g)

    max_inh_g = np.max(inh_g[10000:])
    if max_inh_g > 0.00001:
        axes[row_idx+1, plot_idx].set_ylim(0.0, 1.1*max_inh_g)

for ax1 in axes[:, 1:]:
    for ax in ax1:
        ax.set_xlim(800, 1200)

lines = []
labels = ['sum exc', 'sum inh'] +  plotting_colors["neurons_order"] +  plotting_colors["generators_order"]

gs = axes[2, -1].get_gridspec()

for ax in axes[2:, -1]:
    ax.remove()

legend_axes = fig.add_subplot(gs[2:, -1])

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

legend_axes.legend(lines, labels,  ncol=1, loc='upper left', bbox_to_anchor=(-0.3, 1.0) ) #
legend_axes.axis('off')

for ax0_idx, ax0 in enumerate(axes[:, 0]):
    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)

    if ax0_idx%2 == 0:
        ax0.text(0.0, 0.5, " Возбуждающие \n проводимости", fontsize=TEXTFONTSIZE)
    else:
        ax0.text(0.0, 0.5, " Тормозные \n проводимости", fontsize=TEXTFONTSIZE)

fig.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5, left=0.05, right=0.9)
#fig.tight_layout()

fig.savefig(f'../outputs/plots/{fig_name}.png', dpi=500)
hf.close()
plt.show()