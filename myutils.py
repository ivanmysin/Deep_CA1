import numpy as np
import zipfile
import json
from scipy.signal import hilbert

def get_net_params(filepath):
    with zipfile.ZipFile(filepath, mode='r') as zipped_file:

        config_file = zipped_file.open("config.json", mode='r')
        config = json.loads(config_file.read().decode())

        config_file.close()

    # gen_params = config['config']['layers'][1]['config']['myparams']
    netconfig = config['config']['layers'][2]['config']['cell']['config']
    params = {}
    for key, vals in netconfig.items():

        try:
            v = vals['config']['value']

        except KeyError:
            continue

        except TypeError:
            continue

        params[key] = np.asarray(v).astype(np.float32)

    return params

def get_gen_params(filepath):
    with zipfile.ZipFile(filepath, mode='r') as zipped_file:

        config_file = zipped_file.open("config.json", mode='r')
        config = json.loads(config_file.read().decode())

        config_file.close()

    config_gen_params = config['config']['layers'][1]['config']['myparams']
    return config_gen_params

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

def get_phase_shift(x1, x2, deg=False):
    """
    Рассчитывает средний сдвиг фазы между двумя сигналами с помощью преобразования Гильберта.

    :param x1: первый сигнал (1D array)
    :param x2: второй сигнал (1D array)
    :return: средний сдвиг фазы в радианах
    """
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)

    # Применяем преобразование Гильберта для получения аналитических сигналов
    analytic_x1 = hilbert(x1)
    analytic_x2 = hilbert(x2)

    z = analytic_x1 * np.conj(analytic_x2)

    # Вычисляем разность фаз
    mean_phase_shift = np.angle( np.mean(z), deg=deg )

    return mean_phase_shift