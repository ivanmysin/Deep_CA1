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

dt = 0.1
t = tf.range(0, 120, dt, dtype=tf.float32)
t = tf.reshape(t, shape=(-1, 1))

signal_gens = gen(t)
signal = signal_gens[0, :, :]

signal = tf.cast(signal, dtype=tf.complex64)

omega0 = 6.0
freqs = tf.range(3, 6, 1, dtype=tf.float32)
scales = omega0 / freqs

coeff_map = ()  #  np.zeros((freqs.size, signal.size), dtype=np.complex128)

signal = tf.transpose(signal)

signal_FT = tf.signal.fft(signal)
omegas = fftfreqs(tf.shape(signal)[1], 0.001*dt )
#omegas = tf.stack([omegas, omegas], axis=0)

for idx, s in enumerate(scales):
    morlet_FT = PI**(-0.25) * tf.math.exp(-0.5 * (s * omegas - omega0)**2)
    morlet_FT = tf.cast(morlet_FT, dtype=tf.complex64)


    coeff = tf.signal.ifft(signal_FT * morlet_FT) / tf.cast(tf.math.sqrt(s), dtype=tf.complex64)
    coeff_map = coeff_map + (coeff, )

    # plt.plot(t, tf.transpose( tf.math.real(coeff) ) )
    #
    # plt.show()


print(tf.shape(coeff).numpy())

coeff_map = tf.stack(coeff_map, axis=2)

print(tf.shape(coeff_map).numpy())
signal_pred = tf.reduce_sum(tf.math.real(coeff_map), axis=2)

signal_pred = tf.transpose(signal_pred)

print(tf.shape(signal_pred).numpy())
signal_pred = signal_pred - tf.reduce_min(signal_pred, axis=0, keepdims=True)
# print(tf.shape(signal_pred).numpy())
# l = tf.keras.losses.cosine_similarity(signal, signal_pred)
# print(l)
#
# signal2 =  signal_gens.numpy()[0, :, 1].ravel()
# l = tf.keras.losses.cosine_similarity(signal2, signal_pred)
# print(l)
#
fig, axes = plt.subplots(nrows=3, sharex=True)
#axes.pcolor(t.ravel(), freqs, np.abs(coeff_map), shading='auto')
# plt.colorbar()

axes[0].plot(t, tf.transpose(tf.math.real(signal)) )
axes[1].plot(t, signal_pred[:, 0])
axes[2].plot(t, signal_pred[:, 1])

plt.show()