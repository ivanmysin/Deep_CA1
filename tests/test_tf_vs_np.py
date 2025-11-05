import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, RNN, Layer
import os
os.chdir('../')

from np_meanfield import MeanFieldNetwork as npMeanFieldNetwork
from np_meanfield import SpatialThetaGenerators as npSpatialThetaGenerators
from myutils import get_net_params, get_gen_params

from tensorflow.keras.saving import load_model
from mean_field_class import MeanFieldNetwork, SaveFirings
from genloss import SpatialThetaGenerators, PhaseLockingOutput,  WeightedMSE, WeightedLMSE, FiringsMeanOutRanger

dt = 0.01
duration = 15
model_path = './outputs/big_models/n_deltas_theta_model.keras'

params = get_net_params(model_path)
generator_params = get_gen_params(model_path)

print(params['I_ext'])

# params['gsyn_max'][:] *= 0.0
params['I_ext'][:] *= 1

t = np.arange(0, duration, dt).reshape(1, -1, 1)
tt = tf.convert_to_tensor(t)

generators = SpatialThetaGenerators(generator_params)
firing_generator = generators(tt)
np_firing_generator = firing_generator.numpy()
# k_model = tf.keras.models.load_model(model_path)
k_model = RNN(MeanFieldNetwork(params, dt_dim=dt, use_input=True), return_sequences=True, stateful=True)


np_model = npMeanFieldNetwork(params, dt_dim=dt, use_input=True)

k_firings = k_model(firing_generator)
np_firings = np_model.predict(np_firing_generator)

k_firings = k_firings[0].numpy()

print(k_firings.shape)

print(np.sum( k_firings < 0) )
print(np.sum( np.isnan(k_firings) ) )


# print(np.sum( np_firings[0] < 0))
t = t.reshape(-1)

for i in range(10):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(t, k_firings[:, i], linewidth=3, label='k')
    ax.plot(t, np_firings[0][:, 0, i], linewidth=1, label='np')

    # ax.set_ylim(0, 10)

    ax.legend(loc='upper right')

    plt.show()


