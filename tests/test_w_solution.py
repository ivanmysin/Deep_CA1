import tensorflow as tf
import numpy as np


@tf.function
def vectorized_w_avg_solution(t, a, b, v_avg, r, w_jump, w0):
    """
    Векторизованное решение для w_avg(t) = C1*exp(-a*t) + b*v_avg + r*w_jump/a.

    Параметры:
      t: тензор формы (T,) — точки времени
      a: тензор формы (N,) — параметр a для каждого уравнения
      b: тензор формы (N,)
      v_avg: тензор формы (N,)
      r: тензор формы (N,)
      w_jump: тензор формы (N,)
      w0: тензор формы (N,) — начальное условие w_avg(0)

    Возвращает:
      w_vals: тензор формы (N, T) — w_i(t_j) для i-го уравнения в точке t_j
    """
    # Расширяем t до (1, T) для трансляции
    t_expanded = tf.expand_dims(t, axis=0)  # (1, T)

    # Вычисляем стационарную часть: b*v_avg + (r*w_jump)/a
    stationary_term_1 = b * v_avg  # (N,)

    # Защита от деления на ноль для a
    safe_a = tf.where(tf.abs(a) < 1e-10, tf.ones_like(a), a)  # (N,)
    stationary_term_2 = (r * w_jump) / safe_a  # (N,)

    stationary = stationary_term_1 + stationary_term_2  # (N,)
    stationary = tf.expand_dims(stationary, axis=1)  # (N, 1)

    # C1 = w0 - stationary
    C1 = tf.expand_dims(w0, axis=1) - stationary  # (N, 1)

    # Экспоненциальный множитель: exp(-a*t)
    a_expanded = tf.expand_dims(a, axis=1)  # (N, 1)
    exp_term = tf.exp(-a_expanded * t_expanded)  # (N, T)

    # Итоговое решение
    w_vals = C1 * exp_term + stationary  # (N, T)

    return w_vals


if __name__ == "__main__":
    # Параметры для 3 разных уравнений
    N = 3
    a_vals = tf.constant([0.8, 1.0, 0.5], dtype=tf.float32)           # (3,)
    b_vals = tf.constant([0.2, 0.3, 0.1], dtype=tf.float32)             # (3,)
    v_avg_vals = tf.constant([0.4, 0.6, 0.3], dtype=tf.float32)         # (3,)
    r_vals = tf.constant([1.5, 2.0, 1.0], dtype=tf.float32)              # (3,)
    w_jump_vals = tf.constant([0.7, 0.5, 1.0], dtype=tf.float32)        # (3,)
    w0_vals = tf.constant([0.5, 0.8, 0.4], dtype=tf.float32)            # (3,)  # w_avg(0)

    # Время: 1000 точек от 0 до 10
    t_vals = tf.linspace(0.0, 10.0, 1000)  # (1000,)

    # Вычисляем все решения сразу
    w_matrix = vectorized_w_avg_solution(
        t=t_vals,
        a=a_vals,
        b=b_vals,
        v_avg=v_avg_vals,
        r=r_vals,
        w_jump=w_jump_vals,
        w0=w0_vals
    )  # форма (3, 1000)

    # Конвертируем в NumPy для визуализации
    t_np = t_vals.numpy()
    w_np = w_matrix.numpy()  # (3, 1000)

    # Строим графики для всех 3 уравнений
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(t_np, w_np[i, :], label=f'Уравнение {i+1}', linewidth=2)

    plt.xlabel('Время $t$', fontsize=12)
    plt.ylabel('$w_{\\text{avg}}(t)$', fontsize=12)
    plt.title('Векторизованное решение для $w_{\\text{avg}}(t)$', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
