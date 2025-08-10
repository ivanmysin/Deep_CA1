import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

IZHIKEVICNNEURONSPARAMS = '../parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv'
neurons_params = pd.read_csv(IZHIKEVICNNEURONSPARAMS)
neurons_params.rename( columns={'Izh Vr': 'Vrest', 'Izh Vt': 'Vth', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, inplace=True)

neuron_type = 'CA1 Trilaminar' # 'CA1 Basket' #










# # print(neurons_params.columns)
#

#
# for key in p.columns:
#     print(key, p[key].values)
#
# dt = 0.1
# duration = 500
# t = np.arange(0, duration, dt)
#
# V = np.zeros_like(t)
# U = np.zeros_like(t)
#
# Iext = 200
#
# k = p['k'].iloc[0]
# Vth = p['Vth'].iloc[0]
# Vrest = p['Vrest'].iloc[0]
#
# a = p['a'].iloc[0]
# b = p['b'].iloc[0]
# Cm = p['Cm'].iloc[0]
#
# for i, ts in enumerate(t):
#     if i == 0:
#         V0 = p['Vrest'].iloc[0]
#         U0 = 0
#     else:
#         V0 = V[i - 1]
#         U0 = U[i - 1]
#
#     dVdt = (k * (V0 - Vth) * (V0 - Vrest) - U0  + Iext) / Cm
#     dUdt = a * (b * (V0 - Vrest) - U0 )
#
#     V[i] = V0 + dt * dVdt
#     U[i] = U0 + dt * dUdt
#
#     if V[i] >= p['Vpeak'].iloc[0]:
#         V[i] = p['Vmin'].iloc[0]
#         U[i] = U[i] + p['d'].iloc[0]
#
# fig, axes =  plt.subplots(nrows=2, sharex=True)
# axes[0].plot(t, V)
# axes[1].plot(t, U)
#
#plt.show()