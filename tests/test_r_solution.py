import tensorflow as tf
import numpy as np

@tf.function
def vectorized_solution_r(t, alpha, Delta_eta, v_avg, g_syn_tot, r0):
    """
    Векторизованное решение ОДУ для N уравнений одновременно.

    Параметры:
      t: тензор формы (T,) — точки времени, в которых вычислять r(t)
      alpha: тензор формы (N,) — параметр alpha для каждого уравнения
      Delta_eta: тензор формы (N,)
      v_avg: тензор формы (N,)
      g_syn_tot: тензор формы (N,)
      r0: тензор формы (N,) — начальное условие r(0) для каждого уравнения

    Возвращает:
      r_vals: тензор формы (N, T) — r_i(t_j) для i-го уравнения в точке t_j
    """
    pi = tf.constant(np.pi, dtype=tf.float32)

    # Расширяем t до формы (1, T), чтобы потом транслировать с (N, 1)
    t_expanded = tf.expand_dims(t, axis=0)  # (1, T)

    # Вычисляем показатель экспоненты: shape (N, 1)
    exponent_base = -alpha - g_syn_tot + 2.0 * v_avg
    exponent_base = tf.expand_dims(exponent_base, axis=1)  # (N, 1)

    # Экспонента: exp(t * exponent_base) → shape (N, T)
    exponent = exponent_base * t_expanded
    exp_term = tf.exp(exponent)  # (N, T)

    # Стационарная часть: Delta_eta / (pi * (alpha + g_syn_tot - 2*v_avg))
    denominator = alpha + g_syn_tot - 2.0 * v_avg
    denominator = tf.expand_dims(denominator, axis=1)  # (N, 1)

    # Защита от деления на ноль
    safe_denominator = tf.where(
        tf.abs(denominator) < 1e-10,
        tf.ones_like(denominator),  # временное значение, чтобы не было NaN
        denominator
    )

    stationary = tf.expand_dims(Delta_eta, axis=1) / (pi * safe_denominator)  # (N, 1)

    # C1 = r0 - stationary; r0 уже (N,), расширяем до (N, 1)
    C1 = tf.expand_dims(r0, axis=1) - stationary  # (N, 1)

    # Итоговое решение: C1 * exp_term + stationary
    r_vals = C1 * exp_term + stationary  # (N, T)

    return r_vals

if __name__ == "__main__":
    # Параметры для 3 разных уравнений
    N = 3
    alpha_vals = tf.constant([0.5, 0.3, 0.8], dtype=tf.float32)        # (3,)
    Delta_eta_vals = tf.constant([1.0, 0.8, 1.2], dtype=tf.float32)     # (3,)
    v_avg_vals = tf.constant([0.3, 0.4, 0.2], dtype=tf.float32)          # (3,)
    g_syn_tot_vals = tf.constant([0.2, 0.1, 0.6], dtype=tf.float32)      # (3,)
    r0_vals = tf.constant([0.1, 0.05, 0.15], dtype=tf.float32)           # (3,)

    # Время: 1000 точек от 0 до 10
    t_vals = tf.linspace(0.0, 10.0, 1000)  # (1000,)

    # Вычисляем все решения сразу
    r_matrix = vectorized_solution_r(
        t=t_vals,
        alpha=alpha_vals,
        Delta_eta=Delta_eta_vals,
        v_avg=v_avg_vals,
        g_syn_tot=g_syn_tot_vals,
        r0=r0_vals
    )  # форма (3, 1000)

    # Конвертируем в NumPy для визуализации
    t_np = t_vals.numpy()
    r_np = r_matrix.numpy()  # (3, 1000)

    # Строим графики для всех 3 уравнений
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(t_np, r_np[i, :], label=f'Уравнение {i+1}', linewidth=2)

    plt.xlabel('Время $t$', fontsize=12)
    plt.ylabel('$r(t)$', fontsize=12)
    plt.title('Векторизованное решение для 3 уравнений', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
