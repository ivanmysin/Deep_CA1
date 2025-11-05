import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
os.chdir('../')

from np_meanfield import MeanFieldNetwork as npMeanFieldNetwork
from np_meanfield import SpatialThetaGenerators as npSpatialThetaGenerators
from myutils import get_net_params, get_gen_params

from tensorflow.keras.saving import load_model
from mean_field_class import MeanFieldNetwork, SaveFirings
from genloss import SpatialThetaGenerators, PhaseLockingOutput,  WeightedMSE, WeightedLMSE, FiringsMeanOutRanger

dt = 0.01
duration = 10
model_path = './outputs/big_models/n_deltas_theta_model.keras'

params = get_net_params(model_path)
generator_params = get_gen_params(model_path)

k_model = tf.keras.models.load_model(model_path)

t = np.arange(0, duration, dt).reshape(1, -1, 1)

np_generator = npSpatialThetaGenerators(generator_params)
np_model = npMeanFieldNetwork(params, dt_dim=dt, use_input=True)

k_firings = k_model.predict(t)

firing_generator = np_generator.call(t)
np_firings = np_model.predict(firing_generator)

print(np.sum( k_firings[0] < 0))
print(np.sum( np_firings[0] < 0))
t = t.reshape(-1)

for i in range(10):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(t, k_firings[0][0, :, i], linewidth=3, label='k')
    ax.plot(t, np_firings[0][:, 0, i], linewidth=1, label='np')

    ax.set_ylim(0, 100)

    ax.legend(loc='upper right')

    plt.show()


