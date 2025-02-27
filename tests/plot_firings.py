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

with h5py.File("./firings.h5", mode='r') as h5file:
    firings = h5file['firings'][:]

print(firings.shape)

duration_full_simulation = 1000 * myconfig.TRACK_LENGTH / myconfig.ANIMAL_VELOCITY # ms
t = np.arange(0, duration_full_simulation, myconfig.DT)

genrators = SpatialThetaGenerators(populations[-2:])
firings_generators = genrators(t.reshape(1, -1, 1))
firings_generators = firings_generators.numpy()



firings = np.append(firings, firings_generators, axis=2)

nsubplots = firings.shape[-1]


fig, axes = plt.subplots(nrows=nsubplots, sharex=True, sharey=False)

for f_idx in range(nsubplots):
    axes[f_idx].set_title(populations[f_idx]['type'])
    axes[f_idx].plot(t, firings[0, :, f_idx])

plt.show()