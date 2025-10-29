import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import os
os.chdir('../')
from myutils import get_net_params, get_gen_params

model_path = './outputs/big_models/theta_model.keras'
filepath = 'outputs/firings/units_theta_freq_variation.h5'
filepath_pop = 'outputs/firings/pop_theta_freq_variation.h5'

neurons_params = pd.read_excel('./parameters/neurons_parameters.xlsx', sheet_name='verified_theta_model')
neurons_params['Hippocampome_Neurons_Names'] = neurons_params['Hippocampome_Neurons_Names'].str.strip()
neurons_params['Model_Neurons_Names'] = neurons_params['Model_Neurons_Names'].str.strip()
# neurons_params['Simulated_Type'] = neurons_params['Simulated_Type'].str.strip()
neurons_names = neurons_params[neurons_params['Npops'] == 1]['Model_Neurons_Names'].to_list()

def parzen_filter(arr, window_size=3, axis=0):
    """
    Функция применяет окно Парзена к двумерному массиву вдоль указанной оси.

    :param arr: исходный двумерный массив
    :param window_size: размер окна фильтрации (нечетное число)
    :param axis: ось, вдоль которой выполняется фильтрация (0 — вертикальная, 1 — горизонтальная)
    :return: обработанный массив
    """
    # Проверяем четность окна, оно должно быть нечётным
    if window_size % 2 != 1 or window_size <= 1:
        raise ValueError("Размер окна должен быть положительным нечётным числом.")

    # Центр окна
    center = window_size // 2

    # Создаем ядро окна Парзена
    n = np.arange(-center, center + 1)
    kernel = (1 - (n / center)**2)**2
    kernel /= np.sum(kernel)  # Нормализация ядра

    # Применяем свертку вдоль выбранной оси
    result = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis, arr)

    return result

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

    firingfile.close()

    return v_avg

def get_firings(filepath):
    firingfile = h5py.File(filepath, 'r')

    firings = firingfile['8']['firings'][:]

    firingfile.close()

    return firings

firings_full = get_firings(filepath_pop)
firings_full_units = get_firings(filepath)

firings_full_units = parzen_filter(firings_full_units, window_size=1001, axis=0)

print(firings_full_units.shape)
print(firings_full.shape)

params = get_net_params(model_path)
generators_params = get_gen_params(model_path)

dt = 0.001

gsyn_exc, gsyn_inh = get_gsyn(filepath, params)
# firings_full = firingfile['8']['firings'][:]
gsyn_exc_pop, gsyn_inh_pop = get_gsyn(filepath, params)

v_avg_pop = get_avg(filepath_pop)
v_avg_units = get_avg(filepath)


start_idx = 100000

fig, ax = plt.subplots(nrows=gsyn_exc.shape[1], constrained_layout=True, figsize=(5, 20))


for i in range(gsyn_exc.shape[1]):

    t = np.linspace(0, firings_full.shape[0] * dt, firings_full.shape[0])

    ax[i].set_title(neurons_names[i])
    # ax[i].plot(t[start_idx:], firings_full_units[start_idx:, i], linewidth=1, label='units')
    # ax[i].plot(t[start_idx:], firings_full[start_idx:, i], linewidth=3, label='mean field')


    # ax[i].plot(gsyn_exc[:, i], label='firing rate')
    # ax[i].plot(gsyn_exc[:, i], linewidth=3, label='firing rate')
    # ax[i].plot(gsyn_exc_pop[:, i], linewidth=1, label='firing rate')


    ax[i].plot(t, v_avg_units[:, i], linewidth=1, label='units')
    ax[i].plot(t, v_avg_pop[:, i], linewidth=3, label='mean field')

    ax[i].legend(loc='upper right')

    ax[i].set_xlim(1500, 2500)
    ax[i].set_ylim(-0.2, 1.1)

fig.savefig('./outputs/plots/v_avg.png', dpi=300)


plt.show()