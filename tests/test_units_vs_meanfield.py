import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir('../')

from np_meanfield import MeanFieldNetwork, IzhikevichNetwork, SpatialThetaGenerators
import izhs_lib
import myconfig

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



def make_simulation(params, dt_dim, simtype='meanfield'):
    # model = MeanFieldNetwork(izh_params, dt_dim=dt_dim, use_input=True)

    if simtype == 'meanfield':
        model = MeanFieldNetwork(params, dt_dim=dt_dim, use_input=True)
    elif simtype == 'izh':
        model = IzhikevichNetwork(params, dt_dim=dt_dim, use_input=True)


    rates, hist_states = model.predict(firings_inputs)

    return rates, hist_states

def get_inputs(generator_params, t):
    gens = SpatialThetaGenerators(generator_params)
    # firings_inputs = np.zeros(shape=(1, t.size, Ninps), dtype=np.float32)

    firings_inputs = gens.call(t)

    return firings_inputs

if __name__ == '__main__':
    generator_params = [
        {
            "R": 0.25,
            "OutPlaceFiringRate": 0.5,
            "OutPlaceThetaPhase": 3.14,
            "InPlacePeakRate": 8.0,
            "CenterPlaceField": -5000.0,
            "SigmaPlaceField": 500,
            "SlopePhasePrecession": 0.0, # np.deg2rad(10)*10 * 0.001,
            "PrecessionOnset":  -1.57,
            "ThetaFreq": 8.0,
        },

        {
            "R": 0.5,
            "OutPlaceFiringRate": 5.5,
            "OutPlaceThetaPhase": 0.0,
            "InPlacePeakRate": 18.0,
            "CenterPlaceField": -5000.0,
            "SigmaPlaceField": 500,
            "SlopePhasePrecession": 0.0,  # np.deg2rad(10)*10 * 0.001,
            "PrecessionOnset": np.nan,  # -1.57,
            "ThetaFreq": 8.0,
        },
    ]


    neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
    neurons_params.rename(
        {'Izh Vr': 'Vrest', 'Izh Vt': 'Vth', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd',
         'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin'}, axis=1, inplace=True)

    populations = pd.read_excel('./parameters/neurons_parameters.xlsx', sheet_name='verified_theta_model')
    populations['Hippocampome_Neurons_Names'] = populations['Hippocampome_Neurons_Names'].str.strip()
    populations['Model_Neurons_Names'] = populations['Model_Neurons_Names'].str.strip()
    # neurons_params['Simulated_Type'] = neurons_params['Simulated_Type'].str.strip()
    neurons_names = populations[(populations['Npops'] == 1)&(populations['Simulated_Type'] == 'simulated')]['Hippocampome_Neurons_Names'].to_list()

    NN = 2
    Ninps = len(generator_params)
    dt_dim = 0.001  # ms
    duration = 1000.0
    t = np.arange(0, duration, dt_dim, dtype=np.float32)
    t = t.reshape(1, -1, 1)

    firings_inputs = get_inputs(generator_params, t) #  np.zeros(shape=(1, t.size, Ninps), dtype=np.float32)

    for neuron_name in neurons_names:
        print(neuron_name)
        dim_izh_params = neurons_params[neurons_params["Neuron Type"] == neuron_name]

        dim_izh_params = dim_izh_params.to_dict(orient='records')[0]


        dim_izh_params['Iext'] = 800
        dim_izh_params['V0'] = dim_izh_params['Vrest']
        dim_izh_params['U0'] = 0.0

        # Словарь с константами
        cauchy_dencity_params = {
            'Delta_eta': 50,  # 0.02,
            'bar_eta': 0.0,  # 0.191,
        }

        dim_izh_params = dim_izh_params | cauchy_dencity_params
        izh_params = izhs_lib.dimensional_to_dimensionless(dim_izh_params)
        izh_params['dts_non_dim'] = izhs_lib.transform_T(dt_dim, dim_izh_params['Cm'], dim_izh_params['k'], dim_izh_params['Vrest'])

        for key, val in izh_params.items():
            izh_params[key] = np.zeros(NN, dtype=np.float32) + val

        ## synaptic static variables
        tau_d = 6.02  # ms
        tau_r = 359.8  # ms
        tau_f = 21.0  # ms
        Uinc = 0.25

        gsyn_max = np.zeros(shape=(NN+Ninps, NN), dtype=np.float32)
        gsyn_max[0, 1] = 20
        gsyn_max[1, 0] = 15

        gsyn_max[Ninps:, :] = 5



        pconn = np.zeros(shape=(NN+Ninps, NN), dtype=np.float32)
        pconn[0, 1] = 1
        pconn[1, 0] = 1

        pconn[Ninps:, :] = 1

        Erev = np.zeros(shape=(NN+Ninps, NN), dtype=np.float32) - 75
        Erev[Ninps:, :] = 0.0
        e_r = izhs_lib.transform_e_r(Erev, dim_izh_params['Vrest'])

        izh_params['gsyn_max'] = gsyn_max
        izh_params['pconn'] = pconn
        izh_params['e_r'] = np.zeros_like(gsyn_max) + e_r
        izh_params['tau_d'] = np.zeros_like(gsyn_max) + tau_d
        izh_params['tau_r'] = np.zeros_like(gsyn_max) + tau_r
        izh_params['tau_f'] = np.zeros_like(gsyn_max) + tau_f
        izh_params['Uinc'] = np.zeros_like(gsyn_max) + Uinc




        # for key, val in izh_params.items():
        #     print(key, "\n", val)

        rates_pops, hist_states_pops = make_simulation(izh_params, dt_dim, simtype='meanfield')

        rates_units, hist_states_units = make_simulation(izh_params, dt_dim, simtype='izh')
        rates = parzen_filter(rates_units, window_size=1005, axis=0)

        rates_list = [rates, rates_pops]
        hist_states_list = [hist_states_units, hist_states_pops]

        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 10))
        for i, (rates, hist_states) in enumerate( zip(rates_list, hist_states_list) ):
            #hist_states = hist_states_units
            A = hist_states[-1]

            gsyn_12 = A[:, 0, 1] * gsyn_max[0, 1]
            gsyn_21 = A[:, 1, 0] * gsyn_max[1, 0]

            # print(A.shape)
            # print(rates.shape)
            # print(hist_states[1].shape)

            rates = rates.reshape(-1, NN)
            t = t.ravel()


            if hist_states[1].shape[1] != 1:
                v_avg = np.mean(hist_states[1], axis=2)
                w_avg = np.mean(hist_states[2], axis=2)
            else:
                v_avg = hist_states[1].reshape(-1, NN)
                w_avg = hist_states[2].reshape(-1, NN)


            axes[0].plot(t, rates[:, 0])
            axes[1].plot(t, rates[:, 1])


            axes[2].plot(t, v_avg[:, 0])
            axes[3].plot(t, v_avg[:, 1])


            #axes[2].plot(t, w_avg)


            axes[4].plot(t, gsyn_12)
            axes[5].plot(t, gsyn_21)


        fig.savefig('./outputs/plots/' + neuron_name + '.png', dpi=300, bbox_inches='tight')

        # plt.show(block=False)
        plt.close(fig)

