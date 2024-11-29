import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.chdir("../")
from genloss import SpatialThetaGenerators

params = [
    {
        "R": 0.25,
        "OutPlaceFiringRate": 0.5,
        "OutPlaceThetaPhase": 3.14,
        "InPlacePeakRate": 8.0,
        "CenterPlaceField": 5000.0,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0,  # np.deg2rad(10)*10 * 0.001,
        "PrecessionOnset": np.nan,  # -1.57,
        "ThetaFreq": 8.0,
    },

    {
        "R": 0.25,
        "OutPlaceFiringRate": 0.5,
        "OutPlaceThetaPhase": 3.14,
        "InPlacePeakRate": 8.0,
        "CenterPlaceField": 5000.0,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0,  # np.deg2rad(10)*10 * 0.001,
        "PrecessionOnset": np.nan,  # -1.57,
        "ThetaFreq": 18.0,
    },
]
genrators = SpatialThetaGenerators(params)
t = tf.range(0, 10000.0, 0.1, dtype=tf.float32)
t = tf.reshape(t, shape=(1, -1, 1))

firings = genrators(t)
print(tf.shape(firings))

plt.plot(t[0, :, 0], firings[0, :, 0])
plt.plot(t[0, :, 0], firings[0, :, 1])
plt.show()