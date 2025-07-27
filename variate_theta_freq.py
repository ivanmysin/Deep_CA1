import numpy as np
import myconfig
from myutils import get_net_params, get_gen_params
from np_meanfield import MeanFieldNetwork, SpatialThetaGenerators
import matplotlib.pyplot as plt
import h5py

model_path = './outputs/big_models/base_theta_model.keras'
result_file = './outputs/firings/theta_freq_variation.h5'

firing_file = h5py.File(result_file, mode='w')

params = get_net_params(model_path)
generators_params = get_gen_params(model_path)

dt = myconfig.DT

generators = SpatialThetaGenerators(generators_params)
tnp = np.arange(0, 2500, dt, dtype=np.float32).reshape(1, -1, 1)

generators_firings = generators.call(tnp)

model = MeanFieldNetwork(params, dt_dim=dt, use_input=True)
npfirings, states = model.predict(generators_firings)
initial_states = [s[-1] for s in states]

model.v_threshold = 10000

for theta_freq in range(4, 13):
    generators.set_theta_freq(theta_freq)
    generators_firings = generators.call(tnp)
    npfirings, states = model.predict(generators_firings, initial_states=initial_states)

    #print(npfirings.shape)
    npfirings = npfirings[:, 0, :]

    theta_freq_group = firing_file.create_group(str(theta_freq))

    theta_freq_group.create_dataset(name='firings', data=npfirings)

    theta_freq_group.create_dataset(name='v_avg', data=states[1])
    theta_freq_group.create_dataset(name='w_avg', data=states[2])
    theta_freq_group.create_dataset(name='R', data=states[3])
    theta_freq_group.create_dataset(name='U', data=states[4])
    theta_freq_group.create_dataset(name='A', data=states[5])


firing_file.close()

# generators_firings = generators_firings.reshape(-1, generators_firings.shape[-1])
# tnp = tnp.ravel()
# # plt.plot(tnp, npfirings)
# plt.plot(tnp, generators_firings)
# plt.show()

