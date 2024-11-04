import matplotlib.pyplot as plt
import os

from keras.src.ops import dtype

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
        "ThetaFreq": 6,
        "OutPlaceFiringRate": 8.5,
        "OutPlaceThetaPhase": 0,
        "InPlacePeakRate": 8.0,
        "CenterPlaceField": 2500,
        "R": 0.2,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0,
        "PrecessionOnset": 0.0,
    },
]

gen = SpatialThetaGenerators(params)

dt = 10.0
t = tf.range(0, 1200, dt, dtype=tf.float32)
t = tf.reshape(t, shape=(-1, 1))

signal_gens = gen(t)
signal = signal_gens[0, :, :]

signal = tf.cast(signal, dtype=tf.complex64)

signal = tf.transpose(signal)

signal_FT = tf.signal.fft(signal)

signal_FT = 2.0 * signal_FT / tf.cast(tf.shape(signal)[1], dtype=tf.complex64)

omegas = fftfreqs(tf.shape(signal)[1], 0.001*dt )

omegas = tf.reshape(omegas, shape=(-1, 1))

fig, axes = plt.subplots(nrows=2)
#axes.pcolor(t.ravel(), freqs, np.abs(coeff_map), shading='auto')
# plt.colorbar()

axes[0].plot(t, tf.transpose(tf.math.real(signal)) )
axes[1].plot(omegas, tf.transpose(tf.math.abs(signal_FT)))


plt.show()