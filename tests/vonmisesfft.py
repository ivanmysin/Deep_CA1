import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os


PI = 3.141592653589793

os.chdir("../")
import tensorflow as tf
from genloss import VonMissesGenerator, SpatialThetaGenerators


def fftfreqs(n, dt):
    val = 1.0 / (tf.cast(n, dtype=tf.float32) * dt)
    #results = empty(n, int, device=device)

    N = tf.where( (n%2)==0, n / 2 + 1, (n - 1) / 2)
    p1 = tf.range(0, N, dtype=tf.float32)
    N = tf.where( (n%2)==0, -n/2, -(n-1)/2)
    p2 = tf.range(N, -1, dtype=tf.float32)
    results = tf.concat([p1, p2], axis=0)

    return results * val



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
        "OutPlaceFiringRate" : 0.5,
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
        "OutPlaceFiringRate": 8.5,
        "OutPlaceThetaPhase": 0,
        "InPlacePeakRate": 8.0,
        "CenterPlaceField": -2500,
        "R": 0.2,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0,
        "PrecessionOnset": 0.0,
    },
]

gen = SpatialThetaGenerators(params)

dt = 10.0
t = tf.range(0, 5000, dt, dtype=tf.float32)
t = tf.reshape(t, shape=(-1, 1))

signal_gens = gen(t)
signal_true = signal_gens[0, :, 1]
signal_pred = signal_gens[0, :, 0]

signal_true = signal_true - tf.reduce_mean(signal_true)
signal_pred = signal_pred - tf.reduce_mean(signal_pred)





signal_pred = tf.cast(signal_pred, dtype=tf.complex64)
#signal_pred = tf.transpose(signal_pred)
signal_FT = tf.signal.fft(signal_pred)
# signal_FT = 2.0 * signal_FT / tf.cast(tf.shape(signal)[1], dtype=tf.complex64)
# omegas = fftfreqs(tf.shape(signal)[1], 0.001*dt )
# omegas = tf.reshape(omegas, shape=(-1, 1))

signal_pred_p = tf.signal.ifft(signal_FT**2)
signal_pred_p = tf.cast(signal_pred_p, dtype=tf.float32)


l = tf.keras.losses.cosine_similarity(signal_true, signal_pred_p)
print(l)
fig, axes = plt.subplots(nrows=2)
#axes.pcolor(t.ravel(), freqs, np.abs(coeff_map), shading='auto')
# plt.colorbar()

axes[0].plot(t, signal_pred_p)
axes[1].plot(t, signal_pred)


plt.show()