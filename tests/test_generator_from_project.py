import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
os.chdir("../")
from genloss import SpatialThetaGenerators, PhaseLockingOutput
from pprint import pprint
import pandas as pd

Van = 20 # cm/sec

sheet_name = 'full_local_model'
populations = pd.read_excel('./parameters/neurons_parameters.xlsx', sheet_name=sheet_name)
populations.rename( {'neurons' : 'type'}, axis=1, inplace=True)

populations = populations[populations['Hippocampome_Neurons_Names'] == 'CA1 Pyramidal'][5:10]

populations['SlopePhasePrecession'] = populations['SlopePhasePrecession'] / 360 * 2 * np.pi * Van / 1000

populations['CenterPlaceField'] = populations['CenterPlaceField'] / Van * 1000


print(populations['CenterPlaceField'])

populations['ThetaFreq'] = 8.0
params = populations.to_dict('list')

print(params)



dt = 0.5
duration = 15000
genrators = SpatialThetaGenerators(params)
t = tf.range(0, duration, dt, dtype=tf.float32)
t = tf.reshape(t, shape=(1, -1, 1))

firings = genrators(t)
print(tf.shape(firings))

#
# MeanFirings = [p["OutPlaceFiringRate"] for p in params]
# Rtar = [p["R"] for p in params]
#
# # MeanFirings, ThetaFreq=5.0, dt=0.1
# modulation_layer = PhaseLockingOutput(MeanFirings, ThetaFreq=params[0]['ThetaFreq'], dt=dt)
#
# Rsim = modulation_layer(firings)
#
# print(Rtar)
# print(Rsim.numpy().ravel())
#
#
# df = firings[0, :-1, 0] - firings[0, 1:, 0]

fig, axes = plt.subplots(nrows=1)
axes.plot(t[0, :, 0], firings[0, :, :10])
# axes[1].plot(t[0, 1:, 0],df)
plt.show()