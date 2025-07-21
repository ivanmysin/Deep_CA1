import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
os.chdir("../")
from genloss import SpatialThetaGenerators
from pprint import pprint


neurons_path = './presimulation_files/neurons.pickle'


params = [
    {
        "R": 0.25,
        "OutPlaceFiringRate": 0.5,
        "OutPlaceThetaPhase": 3.14,
        "InPlacePeakRate": 8.0,
        "CenterPlaceField": 5000.0,
        "SigmaPlaceField": 500,
        "SlopePhasePrecession": 0.0, # np.deg2rad(10)*10 * 0.001,
        "PrecessionOnset":  -1.57,
        "ThetaFreq": 8.0,
    },
    #
    # {
    #     "R": 0.25,
    #     "OutPlaceFiringRate": 0.5,
    #     "OutPlaceThetaPhase": 3.14,
    #     "InPlacePeakRate": 8.0,
    #     "CenterPlaceField": 5000.0,
    #     "SigmaPlaceField": 500,
    #     "SlopePhasePrecession": 0.0,  # np.deg2rad(10)*10 * 0.001,
    #     "PrecessionOnset": np.nan,  # -1.57,
    #     "ThetaFreq": 18.0,
    # },
]

# with open(neurons_path, "rb") as neurons_file:
#     populations = pickle.load(neurons_file)
# populations[113]['PrecessionOnset'] = np.nan
# pprint( populations[113])
#
# params = [populations[113], ]
# for pop in populations:
#     if "_generator" in pop['type']:
#         params.append(pop)
#
#         print(pop['CenterPlaceField'])

genrators = SpatialThetaGenerators(params)
t = tf.range(0, 10000, 0.5, dtype=tf.float32)
t = tf.reshape(t, shape=(1, -1, 1))

firings = genrators(t)
print(tf.shape(firings))

df = firings[0, :-1, 0] - firings[0, 1:, 0]

fig, axes = plt.subplots(nrows=2)
axes[0].plot(t[0, :, 0], firings[0, :, 0])
axes[1].plot(t[0, 1:, 0],df)
plt.show()