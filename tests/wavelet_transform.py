import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("../")
import tensorflow as tf
from genloss import VonMissesGenerator, SpatialThetaGenerators

# params = [
#     {
#         "MeanFiringRate" : 12,
#         "R" : 0.3,
#         "ThetaFreq" : 5.0,
#         "ThetaPhase" : 0,
#     },
# ]
#
# gen = VonMissesGenerator(params)
params = [
    {
        "ThetaFreq" : 5,
        "OutPlaceFiringRate" : 15.5,
        "OutPlaceThetaPhase" : 0,
        "InPlacePeakRate" : 8.0,
        "CenterPlaceField" : 2500,
        "R" : 0.3,
        "SigmaPlaceField" : 500,
        "SlopePhasePrecession" : 0.0,
        "PrecessionOnset" : 0.0,
    },
    {
        "ThetaFreq": 5,
        "OutPlaceFiringRate": 15.5,
        "OutPlaceThetaPhase": 0,
        "InPlacePeakRate": 8.0,
        "CenterPlaceField": 2500,
        "R": 0.4,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0,
        "PrecessionOnset": 0.0,
    },
]

gen = SpatialThetaGenerators(params)

t, sampling_period = np.linspace(0, 5, 1000, retstep=True)
#signal = np.cos(2 * np.pi * (50 + 10 * t4signal) * t4signal) + np.cos(2 * np.pi * 40.0 * t4signal)
t = 1000*t.reshape(-1, 1)

signal_gens = gen(t)
signal = signal_gens.numpy()[0, :, 0].ravel()

omega0 = 6.0
freqs = np.arange(5, 25, 5)
scales = omega0 / freqs

coeff_map = np.zeros((freqs.size, signal.size), dtype=np.complex128)

t4signal_FT = np.fft.fft(signal)
omegas = np.fft.fftfreq(t.size, sampling_period)

for idx, s in enumerate(scales):
    morlet_FT = np.pi ** (-0.25) * np.exp(-0.5 * (s * omegas - omega0) ** 2)
    #morlet_FT[omegas < 0] = 0

    coeff = np.fft.ifft(t4signal_FT * morlet_FT) / np.sqrt(s)
    coeff_map[idx, :] = coeff

coeff_map = coeff_map
signal_pred = np.sum(coeff_map.real, axis=0)
signal_pred = signal_pred - np.mean(signal_pred)


signal = signal - np.mean(signal)

l = tf.keras.losses.cosine_similarity(signal, signal_pred)
print(l)

signal2 =  signal_gens.numpy()[0, :, 1].ravel()
signal2 = signal2 - np.mean(signal2)
l = tf.keras.losses.cosine_similarity(signal2, signal_pred)
print(l)

fig, axes = plt.subplots(nrows=2, sharex=True)
#axes.pcolor(t.ravel(), freqs, np.abs(coeff_map), shading='auto')
# plt.colorbar()

axes[0].plot(t, signal)
axes[1].plot(t, signal_pred)

plt.show()