import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import myconfig
import os

os.chdir("../")
from genloss import SpatialThetaGenerators

if myconfig.RUNMODE == 'DEBUG':
    neurons_path = myconfig.STRUCTURESOFNET + "test_neurons.pickle"
    connections_path = myconfig.STRUCTURESOFNET + "test_conns.pickle"
else:
    neurons_path = myconfig.STRUCTURESOFNET + "neurons.pickle"
    connections_path = myconfig.STRUCTURESOFNET + "connections.pickle"

with open(neurons_path, "rb") as neurons_file: ##!!
    populations = pickle.load(neurons_file)

with open(connections_path, "rb") as synapses_file: ##!!
    connections = pickle.load(synapses_file)


with h5py.File(myconfig.OUTPUTSPATH_FIRINGS + "firings_15.h5", mode='r') as h5file:
    firings = h5file['firings'][:]

params4targets_pyrs = []
for pop in populations:
    if pop['type'] == 'CA1 Pyramidal':
        params4targets_pyrs.append(pop)


params4generators = []
for pop in populations:
    if '_generator' in pop['type']:
        params4generators.append(pop)

print(firings.shape)

duration_full_simulation = 2400 #1000 * myconfig.TRACK_LENGTH / myconfig.ANIMAL_VELOCITY # ms
t = np.arange(0, duration_full_simulation, myconfig.DT)

# print(t.shape)
#
# genrators_targents_pyrs = SpatialThetaGenerators(params4targets_pyrs)
# targents_pyrs = genrators_targents_pyrs(t.reshape(1, -1, 1))
# targents_pyrs = targents_pyrs.numpy()
#
# genrators = SpatialThetaGenerators(params4generators)
# generators_firings = genrators(t.reshape(1, -1, 1))
# generators_firings = generators_firings.numpy()
#
#
#
# firings = np.append(firings, generators_firings, axis=2)

for f_idx in range(firings.shape[-1]):

    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=False)


    axes[0].set_title(populations[f_idx]['type'])
    axes[0].plot(t, firings[0, :, f_idx], color='blue')

    print(f_idx, np.sum( np.isnan(firings[0, :, f_idx]) ) )


    # if f_idx < targents_pyrs.shape[-1]:
    #     axes[1].plot(t, targents_pyrs[0, :, f_idx], color='red')

    fig.savefig(myconfig.OUTPUTSPATH_PLOTS + f'{f_idx}.png')

    plt.close(fig)
#plt.show()