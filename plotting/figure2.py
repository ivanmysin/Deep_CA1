import numpy as np
import matplotlib.pyplot as plt
from plots_config import plotting_colors
#from scipy.signal.windows import parzen
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

# Parzen = parzen(101)
# Parzen = Parzen / np.sum(Parzen)

#path = '/home/ivan/Data/phase_relations/!!!LIF_solution_310.hdf5'
path = '/home/ivan/Data/Opt_res/_non_plastic_solution_3263!!!!.hdf5'


hf = h5py.File(path, 'r')
dset_solution = hf['solution']
try:
    dset_targets = hf['targets']
    targets = dset_targets[:]
except KeyError:
    targets = np.zeros_like(dset_solution.shape)

montecarlofile = h5py.File('/home/ivan/Data/interneurons_theta/LIF_Monte_Carlo.hdf5', "r")

t = np.linspace(0, 1800, dset_solution.shape[0])
sine = 0.5 * (np.cos(2 * np.pi * 0.001*t * 7.0) + 1)

gridspec_kw = {
    "width_ratios" : [0.1, 0.9],
}
fig, axes = plt.subplots( nrows=len(plotting_colors["neurons_order"]), ncols=2, \
                          gridspec_kw=gridspec_kw, constrained_layout=True, figsize=(15, 10))


neuron_idxs_in_sol = np.asarray([dset_solution.attrs[neuron_name] for neuron_name in plotting_colors["neurons_order"]])
sourse_neurons_idxes = np.argsort(neuron_idxs_in_sol)

pop_indxes_keys = {}
for neuron_name in plotting_colors["neurons_order"]:
    pop_indxes_keys[neuron_name] = dset_solution.attrs[neuron_name]

for neuron_idx, (neuron_name, neuron_idx_in_sol) in enumerate(sorted(pop_indxes_keys.items(), key=lambda item: item[1])):

    ax2 = axes[neuron_idx, 0]
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(0.5, 0.5, neuron_name, fontsize=TEXTFONTSIZE)

    ax = axes[neuron_idx, 1]
    if neuron_idx == 0:
        ax.set_title("Population frequency")
    if neuron_idx == len(plotting_colors["neurons_order"]) - 1:
        ax.set_xlabel("Time (ms)")
    else:
        ax.xaxis.set_ticklabels([])

    if neuron_idx == int(len(plotting_colors["neurons_order"])//2):
        ax.set_ylabel("sp/sec")



    firings = 1000 * dset_solution[:, neuron_idx_in_sol]
    target = 1000 * dset_targets[:, neuron_idx]
    ax.plot(t, target, label = "target", color='gold', linewidth=4)
    ax.plot(t, firings, color=plotting_colors["neuron_colors"][neuron_name], linewidth=5, label="CBRD")
    neurons_indexes = montecarlofile[neuron_name + "_indexes"][:]
    neurons_times = montecarlofile[neuron_name + "_times"][:]
    montecarlofirings, _ = np.histogram(neurons_times, bins=t)
    montecarlofirings = montecarlofirings / np.max(neurons_indexes + 1)
    montecarlofirings = montecarlofirings / (0.001 * (t[1] - t[0]))
    montecarlofirings = np.convolve(montecarlofirings, Parzen, mode='same')

    #ax.plot(t[:-1], montecarlofirings, linestyle="--", color=plotting_colors["neuron_colors"][neuron_name], label="Monte-Carlo")
    sine_ampls = sine * 0.7*np.max(target)
    ax.plot(t, sine_ampls, linestyle="--", label = "cos", color='black')

    ax.set_ylim(0, 1.1*np.max( [np.max(firings[8000:]), np.max(montecarlofirings[8000:])] )  )



    ax.legend( bbox_to_anchor=(1.2, 1.1), loc="upper right")

    #ax.set_ylim(0, 1.2*np.max(target))
    ax.set_xlim(800, 1800)

fig.savefig('/home/ivan/Data/phase_relations/figures/SFig_2.png', dpi=200)
hf.close()
montecarlofile.close()
plt.show()