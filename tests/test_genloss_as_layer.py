import os
os.chdir("../")
import tensorflow as tf
from genloss import SpatialThetaGenerators
import numpy as np
import matplotlib.pyplot as plt

#############################################################################
default_params = {
            "R": 0.25,
            "OutPlaceFiringRate" : 0.5,
            "OutPlaceThetaPhase" : 2.0, #3.14,
            "InPlacePeakRate" : 8.0,
            "CenterPlaceField" : 5000.0,
            "SigmaPlaceField" : 500,
            "SlopePhasePrecession" : 0.0, #np.deg2rad(10)*10 * 0.001,
            "ThetaFreq" : 8.0,
            "PrecessionOnset" : -1.57,
}

default_params2 = {
            "R": 0.25,
            "OutPlaceFiringRate" : 0.5,
            "OutPlaceThetaPhase" : 2.0, #3.14,
            "InPlacePeakRate" : 8.0,
            "CenterPlaceField" : 5000.0,
            "SigmaPlaceField" : 500,
            "SlopePhasePrecession" : 0.0, #np.deg2rad(10)*10 * 0.001,
            "ThetaFreq" : 8.0,
            "PrecessionOnset" : np.nan,
}

von_mises_params = {
            "R": 0.25,
            "ThetaPhase" : 3.14,
            "ThetaFreq" : 8.0,
            "MeanFiringRate" : 5.0,
}

params = [default_params, default_params2]



t = tf.range(0, 10000.0, 0.1, dtype=tf.float32) #t = tf.constant(0.0, dtype=tf.float32) #

genrators = SpatialThetaGenerators(params)

simulated_firings = genrators(tf.reshape(t, shape=(-1, 1)))


print(simulated_firings)

fig, axes = plt.subplots(nrows=2)
axes[0].plot(t, simulated_firings[0, :, 0])
axes[1].plot(t, simulated_firings[0, :, 1])
plt.show()
