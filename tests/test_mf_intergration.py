import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def solution_r(t, alpha, Delta_eta, v_avg, g_syn_tot, r0):
    """
    Вычисляет r(t) по аналитическому решению ОДУ.

    Параметры:
    - t: тензор времени (может быть скаляром или массивом)
    - alpha, Delta_eta, v_avg, g_syn_tot: параметры уравнения
    - r0: начальное условие r(0)

    Возвращает:
    - r(t): тензор значений функции в точках t
    """
    exp = np.exp
    # Константы
    pi = np.pi #  tf.constant(np.pi, dtype=tf.float32)

    # Показатель экспоненты
    exponent = t * (-alpha - g_syn_tot + 2.0 * v_avg)

    # Стационарная часть (второе слагаемое)
    denominator = alpha + g_syn_tot - 2.0 * v_avg

    #if tf.reduce_any(tf.abs(denominator) < 1e-10):
    if np.any(np.abs(denominator) < 1e-10):
        # Обработка особого случая
        stationary = 0.0  # или другое логичное значение
    else:
        stationary = Delta_eta / (pi * denominator)



    # Определяем C1 из начального условия: r(0) = C1 + stationary = r0
    C1 = r0 - stationary

    # Полное решение
    r_t = C1 * exp(exponent) + stationary
    return r_t

PI = 3.141592653589793
# Определяем переменные
t = sp.symbols('t')
alpha = sp.symbols('alpha')
Delta_eta = sp.symbols('Delta_eta')
#v_avg = sp.symbols('v_avg')
g_syn_tot = sp.symbols('g_syn_tot')
I_ext = sp.symbols('I_ext')
w_avg = sp.symbols('w_avg')
Isyn = sp.symbols('Isyn')
# r = sp.symbols('r')
v_avg = sp.symbols('v_avg')

# Определяем уравнение
r = sp.Function('r')
drdt = Delta_eta / PI + 2.0 * r(t) * v_avg - (alpha + g_syn_tot) * r(t)
equation = sp.Eq(r(t).diff(t), drdt)

#v_avg = sp.Function('v_avg')
# dv/dt = v_avg(t)**2 - alpha * v_avg - w_avg + self.I_ext + Isyn - (PI * rates) ** 2
# dvdt =  v_avg(t)*v_avg(t) - alpha * v_avg(t) - w_avg + I_ext + Isyn - (PI * r)**2
# equation = sp.Eq(v_avg(t).diff(t), dvdt)


# w_avg = w_avg + self.dts_non_dim * (self.a * (self.b * v_avg - w_avg) + self.w_jump * rates)

# # Решаем уравнение
# solution = sp.dsolve(equation)
# print(solution)




# Задаем конкретные значения параметров (пример)
params = {
    alpha: 0.5,
    Delta_eta: 1.0,
    v_avg: 0.3,
    g_syn_tot: 0.2
}

# Подставляем значения параметров в уравнение
equation_subs = equation.subs(params)

# Задаем начальное условие (например, r(0) = 0.1)
initial_condition = {r(0): 0.1}

# Решаем уравнение с начальным условием
solution = sp.dsolve(equation_subs, ics=initial_condition)

# Выводим решение
print("Решение уравнения:")
print(solution)

# Преобразуем символьное решение в численную функцию
r_func = sp.lambdify(t, solution.rhs, 'numpy')

# Создаем массив значений времени для построения графика
t_vals = np.linspace(0, 10, 1000)  # от 0 до 10 с шагом 0.01
r_vals = r_func(t_vals)

r_vals2 = solution_r(t_vals, 0.5, 1.0, 0.3, 0.2, 0.1) # (t, alpha, Delta_eta, v_avg, g_syn_tot, r0)

# Строим график
plt.figure(figsize=(10, 6))
plt.plot(t_vals, r_vals, label='$r(t)$', linewidth=4)
plt.plot(t_vals, r_vals2, label='$r(t)$', linewidth=2)
plt.xlabel('Время $t$', fontsize=12)
plt.ylabel('$r(t)$', fontsize=12)
plt.title('Решение дифференциального уравнения', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

