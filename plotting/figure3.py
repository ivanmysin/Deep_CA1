import numpy as np
import matplotlib.pyplot as plt
from plots_config import plotting_colors
from myutils import get_net_params, get_gen_params
import pandas as pd
from pprint import pprint
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

neurons_params = pd.read_excel('../parameters/neurons_parameters.xlsx', sheet_name='verified_theta_model')
neurons_params['Hippocampome_Neurons_Names'] = neurons_params['Hippocampome_Neurons_Names'].str.strip()
neurons_params['Model_Neurons_Names'] = neurons_params['Model_Neurons_Names'].str.strip()
# neurons_params['Simulated_Type'] = neurons_params['Simulated_Type'].str.strip()
neurons_names = neurons_params[neurons_params['Npops'] == 1]['Model_Neurons_Names'].to_list()
neuron_idx_in_sols = []
for neuron_name in neurons_order:
    neuron_idx_in_sols.append( neurons_names.index(neuron_name)  )


for neuron_name in plotting_colors["generators_order"]:
    neuron_idx_in_sols.append( neurons_names.index(neuron_name) )

model_path = '../outputs/big_models/theta_model.keras'

params = get_net_params(model_path)


path_sim = '../outputs/firings/theta_freq_variation.h5'
hf = h5py.File(path_sim, 'r')

gsyn_max = params["gsyn_max"]

Afull = hf['8']['A'][:]
gsyn_full = Afull * gsyn_max

is_exc = params['e_r'] > 0
is_inh = params['e_r'] < 0
gsyn_total = np.sum(gsyn_full, axis=1)

gsyn_exc = np.sum(gsyn_full * is_exc, axis=1)
gsyn_inh = np.sum(gsyn_full * is_inh, axis=1)

hf.close()


gridspec_kw = {
    "width_ratios" : [0.15, ] + [0.1, ] * 2 + [0.1, ]
}

t = np.linspace(0, duration, int(duration / dt) )
sine = 0.5 * (np.cos(2 * np.pi * 0.001*t * 8.0) + 1)


nrows = 10
fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(20, 15), gridspec_kw=gridspec_kw)




for neuron_idx, neuron_name in enumerate(neurons_order):

    plot_idx = neuron_idx%2 + 1

    row_idx = 2 * int(neuron_idx / 2)



    neuron_name_title = neuron_name
    # if len(neuron_name_title) > 12:
    #     neuron_name_title = neuron_name_title[:5] +  neuron_name_title[5:].replace(" ", "\n", 1)


    axes[row_idx, plot_idx].set_title(neuron_name_title)

    # if neuron_idx == 0 or neuron_idx == 4:
    #     axes[row_idx, plot_idx].set_ylabel(r"")
    #     axes[row_idx+1, plot_idx].set_ylabel(r"")

    if row_idx == (nrows - 2) or neuron_idx == len(neurons_order) - 1:
        axes[row_idx+1, plot_idx].set_xlabel("Время (мс)")
    else:
        axes[row_idx+1, plot_idx].xaxis.set_ticklabels([])

    axes[row_idx, plot_idx].xaxis.set_ticklabels([])



    for pre_idx, pre_name in enumerate(neurons_names):


        if not ((pre_name in plotting_colors["neurons_order"]) or (pre_name in plotting_colors["generators_order"])):
            continue

        g_syn = gsyn_full[:, neuron_idx_in_sols[pre_idx], neuron_idx_in_sols[neuron_idx]]


        if pre_name in ["Pyramidal (deep)", "Pyramidal (superficial)" , "CA3 Input", "MEC Input"] :
            ax = axes[row_idx, plot_idx]
        else:
            ax = axes[row_idx+1, plot_idx]


        color = plotting_colors["neuron_colors"][pre_name]
        ax.plot(t, g_syn, linestyle="-", label=pre_name, color=color)


    axes[row_idx, plot_idx].plot(t, gsyn_exc[:, neuron_idx_in_sols[neuron_idx]], linestyle=(0, (1, 1)), label="sum exc", color='orange', linewidth=2)
    axes[row_idx+1, plot_idx].plot(t, gsyn_inh[:, neuron_idx_in_sols[neuron_idx]], linestyle=(0, (1, 1)), label="sum inh", color='magenta', linewidth=2)

    sine_amples_exc = 0.7*np.max(gsyn_exc[:, neuron_idx_in_sols[neuron_idx]]) * sine
    axes[row_idx, plot_idx].plot(t, sine_amples_exc, linestyle='--', label="cos", color='black')

    sine_amples_inh = 0.7*np.max(gsyn_inh[:, neuron_idx_in_sols[neuron_idx]]) * sine
    axes[row_idx+1, plot_idx].plot(t, sine_amples_inh, linestyle="--", label="cos", color='black')

    max_exc_g = np.max(gsyn_exc[:, neuron_idx_in_sols[neuron_idx]])
    if max_exc_g > 0.00001:
        axes[row_idx, plot_idx].set_ylim(0.0, 1.1*max_exc_g)

    max_inh_g = np.max(gsyn_inh[:, neuron_idx_in_sols[neuron_idx]])
    if max_inh_g > 0.00001:
        axes[row_idx+1, plot_idx].set_ylim(0.0, 1.1*max_inh_g)

for ax1 in axes[:, 1:]:
    for ax in ax1:
        ax.set_xlim(800, 1200)

        ax.get_yaxis().get_major_formatter().set_scientific(False)

lines = []
labels = ['sum exc', 'sum inh'] +  plotting_colors["neurons_order"] +  plotting_colors["generators_order"]

gs = axes[2, -1].get_gridspec()

for ax in axes[:, -1]:
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

plt.show()

