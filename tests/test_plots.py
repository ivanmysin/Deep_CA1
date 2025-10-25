import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

os.chdir('../')
from myutils import get_net_params, get_gen_params
model_path = './outputs/big_models/theta_model.keras'
filepath = 'outputs/firings/units_theta_freq_variation.h5'
filepath_pop = 'outputs/firings/pop_theta_freq_variation.h5'

def get_gsyn(filepath, params):
    firingfile = h5py.File(filepath, 'r')

    Afull = firingfile['8']['A'][:]

    gsyn = Afull * params['gsyn_max']

    is_exc = (params['e_r'] > 0).astype(np.float32)
    is_inh = (params['e_r'] < 0).astype(np.float32)

    gsyn_exc = np.sum(gsyn * is_exc, axis=1)
    gsyn_inh = np.sum(gsyn * is_inh, axis=1)

    firingfile.close()

    return gsyn_exc, gsyn_inh

def get_avg(filepath):
    firingfile = h5py.File(filepath, 'r')

    v_avg = firingfile['8']['v_avg'][:]



    # print(v_avg.shape)
    #
    # if v_avg.shape[1] == 1:
    #     v_avg = v_avg[:, 0, :]
    # else:
    #     v_avg = np.mean(v_avg, axis=2)

    firingfile.close()

    return v_avg
params = get_net_params(model_path)
generators_params = get_gen_params(model_path)

dt = 0.01

gsyn_exc, gsyn_inh = get_gsyn(filepath, params)
# firings_full = firingfile['8']['firings'][:]
gsyn_exc_pop, gsyn_inh_pop = get_gsyn(filepath, params)

v_avg_pop = get_avg(filepath_pop)
v_avg_units = get_avg(filepath)


t = np.linspace(0, gsyn_exc.shape[0] * dt, gsyn_exc.shape[0])




fig, ax = plt.subplots(nrows=gsyn_exc.shape[1])
for i in range(gsyn_exc.shape[1]):


    # ax[i].plot(t, firings_full[:, i], label='firing rate')
    # ax[i].plot(gsyn_exc[:, i], label='firing rate')
    # ax[i].plot(gsyn_exc[:, i], linewidth=3, label='firing rate')
    # ax[i].plot(gsyn_exc_pop[:, i], linewidth=1, label='firing rate')


    ax[i].plot(v_avg_units[:, i], linewidth=1, label='units')
    ax[i].plot(v_avg_pop[:, i], linewidth=3, label='mean field')

    ax[i].legend(loc='upper right')






plt.show()