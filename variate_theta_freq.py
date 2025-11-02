import numpy as np
import myconfig
from myutils import get_net_params, get_gen_params
from np_meanfield import MeanFieldNetwork, SpatialThetaGenerators, IzhikevichNetwork
import matplotlib.pyplot as plt
import h5py

DT_COEFF = 0.1


def make_simulation(params, result_file, simulation_type):
    dt = myconfig.DT * DT_COEFF  # шаг в мс
    duration = 2500  # время симуляции в мс
    save_interval = 20  # интервал сохранения в мс

    generators = SpatialThetaGenerators(generators_params)
    tnp = np.arange(0, duration, dt, dtype=np.float32).reshape(1, -1, 1)

    generators_firings = generators.call(tnp)

    if simulation_type == 'units':
        model = IzhikevichNetwork(params, dt_dim=dt, use_input=True)

    elif simulation_type == 'meanfield':
        model = MeanFieldNetwork(params, dt_dim=dt, use_input=True)
        model.NN = 1
        model.Npops = 10
    else:
        raise('Invalid simulation_type')

    # Вычисляем количество шагов для сохранения
    n_steps_total = int(duration / dt)
    n_steps_per_save = int(save_interval / dt)
    n_saves = int(duration / save_interval)

    # Первоначальная симуляция для получения начальных состояний
    npfirings, states = model.predict(generators_firings[:, :100, :], save_states=False)
    initial_states = [s[-1] for s in states]

    firing_file = h5py.File(result_file, mode='w')

    for theta_freq in [8, ]:  # range(4, 13):
        generators.set_theta_freq(theta_freq)
        generators_firings = generators.call(tnp)

        theta_freq_group = firing_file.create_group(str(theta_freq))

        # Создаем datasets для хранения результатов
        firings_ds = theta_freq_group.create_dataset(
            name='firings',
            shape=(n_steps_total, model.Npops),  # форма: (все_шаги, units)
            dtype=np.float32
        )
        v_avg_ds = theta_freq_group.create_dataset(
            name='v_avg',
            shape=(n_steps_total, model.Npops),
            dtype=np.float32
        )
        w_avg_ds = theta_freq_group.create_dataset(
            name='w_avg',
            shape=(n_steps_total, model.Npops),
            dtype=np.float32
        )
        R_ds = theta_freq_group.create_dataset(
            name='R',
            shape=(n_steps_total,) + states[3].shape[1:],
            dtype=np.float32
        )
        U_ds = theta_freq_group.create_dataset(
            name='U',
            shape=(n_steps_total,) + states[4].shape[1:],
            dtype=np.float32
        )
        A_ds = theta_freq_group.create_dataset(
            name='A',
            shape=(n_steps_total,) + states[5].shape[1:],
            dtype=np.float32
        )

        current_states = initial_states.copy()
        current_time_index = 0

        # Разбиваем симуляцию на блоки по save_interval
        for save_idx in range(n_saves):
            start_step = save_idx * n_steps_per_save
            end_step = (save_idx + 1) * n_steps_per_save

            # Выбираем часть входных данных для текущего интервала
            block_input = generators_firings[:, start_step:end_step, :]

            # Запускаем симуляцию для текущего блока
            npfirings_block, states_block = model.predict(
                block_input,
                initial_states=current_states,
                save_states=True
            )

            if states_block[1].shape[1] != 1:
                v_avg_block = np.mean(states_block[1], axis=2)
                w_avg_block = np.mean(states_block[2], axis=2)
            else:
                v_avg_block = states_block[1][:, 0, :]
                w_avg_block = states_block[2][:, 0, :]
            # Сохраняем результаты текущего блока
            firings_ds[current_time_index:current_time_index + n_steps_per_save] = npfirings_block[:, 0, :]
            v_avg_ds[current_time_index:current_time_index + n_steps_per_save] = v_avg_block  # states_block[1]
            w_avg_ds[current_time_index:current_time_index + n_steps_per_save] = w_avg_block  # states_block[2]
            R_ds[current_time_index:current_time_index + n_steps_per_save] = states_block[3]
            U_ds[current_time_index:current_time_index + n_steps_per_save] = states_block[4]
            A_ds[current_time_index:current_time_index + n_steps_per_save] = states_block[5]

            # Обновляем состояния для следующего блока
            current_states = [s[-1] for s in states_block]
            current_time_index += n_steps_per_save

            print(f'Theta freq {theta_freq}: saved block {save_idx + 1}/{n_saves}')

        # Обновляем начальные состояния для следующей частоты
        initial_states = current_states
        print(f'{theta_freq} is simulated')

    firing_file.close()
#########################################################################
model_path = './outputs/big_models/theta_model.keras' # '/home/ivanmysin/nice_theta_models/theta_model_5000.keras'   #
result_file_units = './outputs/firings/units_theta_freq_variation.h5'
result_file_pop = './outputs/firings/pop_theta_freq_variation.h5'


params = get_net_params(model_path)
generators_params = get_gen_params(model_path)

# params['pconn'][:-4, :] = 0.0 # отключаем все тормозные связи
params['dts_non_dim'][:] *= DT_COEFF

# make_simulation(params, result_file_pop, 'meanfield')
make_simulation(params, result_file_units, 'units')


# params['v_peak'] = np.zeros((10, 10), dtype=np.float32) + 300
# params['v_reset'] = np.zeros((10, 10), dtype=np.float32) - 300


# generators_firings = generators_firings.reshape(-1, generators_firings.shape[-1])

# npfirings = npfirings.reshape(-1, npfirings.shape[-1])
# tnp = tnp.ravel()
# plt.plot(tnp, npfirings)
# ## plt.plot(tnp, generators_firings)
# plt.show()

