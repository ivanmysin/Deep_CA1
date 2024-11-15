import numpy as np
import matplotlib.pyplot as plt
import h5py

filepath = "../population_datasets/CA1 Basket CCK+/0.hdf5"

with h5py.File(filepath, "r") as h5file:
    firings = h5file["firing_rate"][:]



normal_firings = np.log(firings + 1.0)
#normal_firings = (firings**3 - 1) / 3

plt.hist(normal_firings, bins=200, density=True)
plt.show()

# firings_reverse = np.exp(normal_firings) - 1.0
#
# plt.plot(firings)
# plt.plot(firings_reverse)
# plt.show()