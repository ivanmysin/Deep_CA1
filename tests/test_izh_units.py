import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir("../")
import myconfig
SimulatedType = "CA1 Horizontal Axo-Axonic"  #"CA1 Basket" #

neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
neurons_params = neurons_params.fillna(-1)
neurons_params = neurons_params.rename(columns={'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd', 'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin',})
neurons_params = neurons_params.drop(['CARLsim_default', 'E/I', 'Population Size', 'Refractory Period', 'rank'], axis=1)
tp = neurons_params.loc[ neurons_params["Neuron Type"] == SimulatedType]

Vrest = tp["Vrest"].values[0]
Vth = tp["Vth_mean"].values[0]
Vpeak = tp["Vpeak"].values[0]
Vmin = tp["Vmin"].values[0]
Cm = tp["Cm"].values[0]
a = tp["a"].values[0]
b = tp["b"].values[0]
d = tp["d"].values[0]
d = tp["d"].values[0]
k = tp["k"].values[0]
dt = 0.1
V = np.zeros(20000, dtype=np.float64)
V[0] = Vrest
U = np.zeros_like(V)
Iinh = 0
Iexc = 200.2

for i in range(1, V.size):
    dV = dt * (k * (V[i - 1] - Vrest) * (V[i - 1] - Vth) - U[i - 1] + Iexc + Iinh) / Cm #+ sigma * xi / ms ** 0.5: volt
    dU = dt * a * (b * (V[i - 1] - Vrest) - U[i - 1])

    V[i] = V[i - 1] + dV
    U[i] = U[i - 1] + dU

    if V[i] > Vpeak:
        V[i] = Vmin
        U[i] = U[i] + d

fig, axes = plt.subplots(nrows=2)
axes[0].plot(V)
axes[1].plot(U)
plt.show()

# Iexc = gexc * (Eexc - V): ampere
# Iinh = ginh * (Einh - V): ampere
# gexc = ampl_1_e * 0.5 * (cos(2 * pi * t * omega_1_e + phase0_1_e) + 1) + ampl_2_e * 0.5 * (
#             cos(2 * pi * t * omega_2_e + phase0_2_e) + 1) + ampl_3_e * 0.5 * (
#                    cos(2 * pi * t * omega_3_e + phase0_3_e) + 1) + ampl_4_e * 0.5 * (
#                    cos(2 * pi * t * omega_4_e + phase0_4_e) + 1): siemens
# ginh = ampl_1_i * 0.5 * (cos(2 * pi * t * omega_1_i + phase0_1_i) + 1) + ampl_2_i * 0.5 * (
#             cos(2 * pi * t * omega_2_i + phase0_2_i) + 1) + ampl_3_i * 0.5 * (
#                    cos(2 * pi * t * omega_3_i + phase0_3_i) + 1) + ampl_4_i * 0.5 * (
#                    cos(2 * pi * t * omega_4_i + phase0_4_i) + 1): siemens
# Vth: volt