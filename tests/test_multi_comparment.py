import numpy as np

class MultiCompartmentIzhikevich:
    def __init__(self, parameters):
        """
        Многокомпартментная модель нейрона Ижикевича

        Parameters:
        parameters (dict): Словарь с параметрами модели
        """
        self.params = parameters
        self.n_compartments = self._detect_compartments()
        self.V = None
        self.U = None
        self._initialize_state_variables()
        self._setup_coupling_matrix()

    def _detect_compartments(self):
        """Определяет количество компартментов по переданным параметрам"""
        max_comp = 0
        for key in self.params.keys():
            if key.startswith('k') and key[-1].isdigit():
                comp_num = int(key[-1])
                max_comp = max(max_comp, comp_num)
        return max_comp + 1  # +1 потому что нумерация с 0

    def _initialize_state_variables(self):
        """Инициализирует переменные состояния"""
        # Начальные значения - потенциал покоя для каждого компартмента
        self.V = np.array([self.params.get(f'Vr{i}', -65.0) for i in range(self.n_compartments)])
        self.U = np.zeros(self.n_compartments)

    def _setup_coupling_matrix(self):
        """Создает матрицу связи между компартментами через свертку"""
        # Инициализируем матрицу проводимостей
        self.G_matrix = np.zeros((self.n_compartments, self.n_compartments))
        self.P_matrix = np.zeros((self.n_compartments, self.n_compartments))

        # Заполняем матрицы параметрами связи
        for i in range(self.n_compartments - 1):
            G_key = f'G{i}'
            P_key = f'P{i}'
            if G_key in self.params and P_key in self.params:
                G = self.params[G_key]
                P = self.params[P_key]

                # Проксимальный к дистальному
                self.G_matrix[i, i+1] = G * P
                self.G_matrix[i+1, i] = G * (1 - P)

                # Сохраняем P для асимметрии
                self.P_matrix[i, i+1] = P
                self.P_matrix[i+1, i] = 1 - P

        # Создаем ядро свертки для пространственной производной
        # Для линейной цепочки компартментов используем оператор Лапласа

        self.G_matrix *= self.P_matrix
        self.laplacian_kernel = np.array([1, -2, 1])

    def _calculate_coupling_currents_convolution(self):
        """Вычисляет токи связи через свертку по пространственной координате"""
        if self.n_compartments == 1:
            return np.zeros(1)

        # Вычисляем вторую пространственную производную потенциала
        # Используем дискретный оператор Лапласа
        laplacian_V = np.zeros(self.n_compartments)

        # Внутренние точки
        for i in range(1, self.n_compartments - 1):
            laplacian_V[i] = self.V[i-1] - 2 * self.V[i] + self.V[i+1]

        # Граничные условия (Неймана - нулевой поток на границах)
        if self.n_compartments > 1:
            laplacian_V[0] = self.V[1] - self.V[0]  # первая разность
            laplacian_V[-1] = self.V[-2] - self.V[-1]  # первая разность

        # Применяем матрицу проводимостей к пространственной производной
        coupling_currents = np.zeros(self.n_compartments)

        for i in range(self.n_compartments):
            for j in range(self.n_compartments):
                if i != j and self.G_matrix[i, j] > 0:
                    # Ток пропорционален разности потенциалов и проводимости
                    coupling_currents[i] += self.G_matrix[i, j] * (self.V[j] - self.V[i])

        return coupling_currents

    def _calculate_coupling_currents_laplacian(self):
        """Альтернативный метод: использует дискретный оператор Лапласа"""
        if self.n_compartments == 1:
            return np.zeros(1)

        coupling_currents = np.zeros(self.n_compartments)

        # Для каждого компартмента вычисляем сумму токов от соседей
        for i in range(self.n_compartments):
            # Левый сосед (если существует)
            if i > 0:
                G_left = self.G_matrix[i, i-1]
                if G_left > 0:
                    coupling_currents[i] -= G_left * (self.V[i] - self.V[i-1])

            # Правый сосед (если существует)
            if i < self.n_compartments - 1:
                G_right = self.G_matrix[i, i+1]
                if G_right > 0:
                    coupling_currents[i] -= G_right * (self.V[i] - self.V[i+1])

        return coupling_currents

    def _get_compartment_params(self, comp_index):
        """Возвращает параметры для указанного компартмента"""
        prefix = '' if comp_index == 0 else str(comp_index)
        return {
            'k': self.params.get(f'k{prefix}', 0.04),
            'a': self.params.get(f'a{prefix}', 0.02),
            'b': self.params.get(f'b{prefix}', 0.2),
            'd': self.params.get(f'd{prefix}', 8.0),
            'C': self.params.get(f'C{prefix}', 100.0),
            'Vr': self.params.get(f'Vr{prefix}', -65.0),
            'Vt': self.params.get(f'Vt{prefix}', -50.0),
            'Vpeak': self.params.get(f'Vpeak{prefix}', 35.0),
            'Vmin': self.params.get(f'Vmin{prefix}', -65.0)
        }

    def simulate(self, I_inj, dt=0.1, duration=1000, method='laplacian'):
        """
        Запускает симуляцию модели

        Parameters:
        I_inj: Входной ток
        dt: Шаг по времени (мс)
        duration: Длительность симуляции (мс)
        method: Метод расчета токов ('laplacian' или 'convolution')

        Returns:
        dict: Словарь с результатами симуляции
        """
        steps = int(duration / dt)

        # Подготовка входного тока
        if np.isscalar(I_inj):
            I_inj = np.full((steps, self.n_compartments), I_inj)
        elif I_inj.ndim == 1:
            I_inj_full = np.zeros((steps, self.n_compartments))
            I_inj_full[:, 0] = I_inj[:steps]
            I_inj = I_inj_full
        elif I_inj.shape[1] != self.n_compartments:
            raise ValueError("Размерность I_inj не соответствует количеству компартментов")

        # Массивы для хранения результатов
        V_history = np.zeros((steps, self.n_compartments))
        U_history = np.zeros((steps, self.n_compartments))
        spike_times = [[] for _ in range(self.n_compartments)]

        # Инициализация
        self._initialize_state_variables()

        # Выбор метода расчета токов
        if method == 'convolution':
            calc_coupling = self._calculate_coupling_currents_convolution
        else:
            calc_coupling = self._calculate_coupling_currents_laplacian

        # Основной цикл симуляции
        for t in range(steps):
            # Сохраняем текущие состояния
            V_history[t] = self.V.copy()
            U_history[t] = self.U.copy()

            # Вычисляем токи связи
            I_coupling = calc_coupling()

            # Обновляем каждый компартмент
            for i in range(self.n_compartments):
                p = self._get_compartment_params(i)

                # Проверяем условие спайка
                if self.V[i] >= p['Vpeak']:
                    self.V[i] = p['Vmin']
                    self.U[i] += p['d']
                    spike_times[i].append(t * dt)

                # Вычисляем производные
                dV_dt = (p['k'] * (self.V[i] - p['Vr']) * (self.V[i] - p['Vt']) -
                         self.U[i] + I_inj[t, i] + I_coupling[i]) / p['C']
                dU_dt = p['a'] * (p['b'] * (self.V[i] - p['Vr']) - self.U[i])

                # Интегрируем методом Эйлера
                self.V[i] += dV_dt * dt
                self.U[i] += dU_dt * dt

        return {
            'V': V_history,
            'U': U_history,
            'spike_times': spike_times,
            't': np.arange(0, duration, dt),
            'n_compartments': self.n_compartments,
            'coupling_method': method
        }

    def get_coupling_info(self):
        """Возвращает информацию о связях между компартментами"""
        info = {
            'G_matrix': self.G_matrix,
            'P_matrix': self.P_matrix,
            'n_compartments': self.n_compartments
        }

        # Информация о конкретных связях
        connections = []
        for i in range(self.n_compartments - 1):
            G_key = f'G{i}'
            P_key = f'P{i}'
            if G_key in self.params and P_key in self.params:
                connections.append({
                    'from': i,
                    'to': i+1,
                    'G': self.params[G_key],
                    'P': self.params[P_key],
                    'G_proximal': self.G_matrix[i, i+1],  # от i к i+1
                    'G_distal': self.G_matrix[i+1, i]     # от i+1 к i
                })

        info['connections'] = connections
        return info

# Пример использования с визуализацией связей
if __name__ == "__main__":
    # Параметры для 3-компартментной модели
    params_3comp = {
        'k0': 0.04, 'a0': 0.02, 'b0': 0.2, 'd0': 8.0, 'C0': 100.0,
        'Vr0': -65.0, 'Vt0': -50.0, 'Vpeak0': 35.0, 'Vmin0': -65.0,

        'k1': 0.04, 'a1': 0.02, 'b1': 0.2, 'd1': 6.0, 'C1': 100.0,
        'Vr1': -65.0, 'Vt1': -50.0, 'Vpeak1': 35.0, 'Vmin1': -65.0,

        'k2': 0.04, 'a2': 0.02, 'b2': 0.2, 'd2': 4.0, 'C2': 100.0,
        'Vr2': -65.0, 'Vt2': -50.0, 'Vpeak2': 35.0, 'Vmin2': -65.0,

        'G0': 0.1, 'P0': 0.5,
        'G1': 0.1, 'P1': 0.5,
    }

    # Создание и симуляция модели
    neuron = MultiCompartmentIzhikevich(params_3comp)
    print(f"Количество компартментов: {neuron.n_compartments}")

    # Информация о связях
    coupling_info = neuron.get_coupling_info()
    print("Матрица проводимостей:")
    print(coupling_info['G_matrix'])

    # Входной ток только для сомы (компартмент 0)
    I_inj = np.zeros(10000)
    I_inj[1000:9000] = 15.0  # Ток 15 пА

    # Сравнение методов
    results_laplacian = neuron.simulate(I_inj, dt=0.1, duration=1000, method='laplacian')
    results_convolution = neuron.simulate(I_inj, dt=0.1, duration=1000, method='convolution')

    # Визуализация результатов
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for i in range(neuron.n_compartments):
        axes[0].plot(results_laplacian['t'], results_laplacian['V'][:, i],
                     label=f'Компартмент {i} (Laplacian)')
        axes[1].plot(results_convolution['t'], results_convolution['V'][:, i],
                     label=f'Компартмент {i} (Convolution)')

    axes[0].set_ylabel('Мембранный потенциал (мВ)')
    axes[0].set_title('Метод Лапласа')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_ylabel('Мембранный потенциал (мВ)')
    axes[1].set_xlabel('Время (мс)')
    axes[1].set_title('Метод свертки')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()