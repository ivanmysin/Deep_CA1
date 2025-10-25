import numpy as np
import matplotlib.pyplot as plt

Npops = 10
NN = 1000

eta_bar = np.zeros(Npops).reshape(-1, 1)
Delta_eta = np.ones(Npops).reshape(-1, 1)

rads = np.random.uniform(-np.pi, np.pi, int(NN * Npops)).reshape(Npops, NN)
eta = eta_bar + Delta_eta * np.tan( rads )

print(eta.shape)
# # eta = cauchy.rvs(loc=eta_bar, scale=Delta_eta, size=100000)
#
#
# eta_hist, eta_bins = np.histogram(eta, bins='auto', density=True, range=(-200, 200))
# eta_bins = (eta_bins[1:] + eta_bins[:-1]) / 2.0
#
# print('eta_bins', eta_bins.size)
# print('eta', eta_bins.min(), eta_bins.max())
#
#
# eta_range = np.linspace(-200, 200, 1000)
# lorenz = Delta_eta / (np.pi * ((eta_range - eta_bar)**2 + Delta_eta**2))
#
#
# plt.plot(eta_bins, eta_hist)
# plt.plot(eta_range, lorenz)
# plt.show()
#
