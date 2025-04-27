import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sympy.printing.pretty.pretty_symbology import line_width


def runge_kutta_4(f, y0, t0, h):
    """
    Реализует метод Рунге-Кутта 4-го порядка для решения дифференциального уравнения dy/dt = f(t,y).

    :param f: функция, задающая правую часть уравнения dy/dt = f(t,y)
    :param y0: начальное значение y(t0)
    :param t0: начальная точка времени
    :param h: шаг интегрирования
    :return: новое приближенное значение y(t+h)
    """
    k1 = h * f(t0, y0)
    k2 = h * f(t0 + h / 2, y0 + k1 / 2)
    k3 = h * f(t0 + h / 2, y0 + k2 / 2)
    k4 = h * f(t0 + h, y0 + k3)

    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate(func, y0, t0, dt, duration):
    ys = []

    for t in np.arange(t0, duration, dt):
        y = runge_kutta_4(func, y0, t, dt)

        y0 = y

        ys.append(y)

    return np.stack(ys, axis=1)


class SimplestMeanField:

    def __init__(self, alpha, a, b, w_jump, dts_non_dim, Delta_eta, I_ext):
        self.alpha = alpha
        self.a = a
        self.b = b
        self.w_jump = w_jump
        self.dts_non_dim = dts_non_dim
        self.Delta_eta = Delta_eta

        self.I_ext = I_ext

    def __call__(self, t, state):
        rates = state[0]
        v_avg = state[1]
        w_avg = state[2]
        g_syn_tot = 0.0
        Isyn = 0.0

        drdt = self.Delta_eta / np.pi + 2 * rates * v_avg - (self.alpha + g_syn_tot) * rates
        dvdt = v_avg ** 2 - self.alpha * v_avg - w_avg + self.I_ext + Isyn - (np.pi * rates) ** 2
        dwdt = self.a * (self.b * v_avg - w_avg) + self.w_jump * rates


        return np.asarray([drdt, dvdt, dwdt])
#####################################################################
dt = 0.5 ##
duration = 200

alpha = 0.38348082595870203
a = 0.008311497425623033
b = 0.003207946374801873
Delta_eta = 0.02024164418659393
dt_non_dim = 0.30078815789473684 # 0.06015763157894737
w_jump = 0.0005060411046648482
I_ext = 0.17711438663269688
mean_field = SimplestMeanField(alpha, a, b, w_jump, dt_non_dim, Delta_eta, I_ext)

y0 = [0.0, 0.0, 0.0]
t = np.arange(0, duration, dt)
duration_non_dim = duration / dt * dt_non_dim

t_non_dim = np.arange(0, duration_non_dim, dt_non_dim)
sol = solve_ivp(mean_field, [0, duration_non_dim], y0, method='RK45', t_eval=t_non_dim)


mysol = simulate(mean_field, y0, 0.0, dt_non_dim, duration_non_dim)

fig, axes = plt.subplots(nrows=3)

axes[0].plot(t, sol.y[0, :], linewidth=5)
axes[0].plot(t, mysol[0, :], linewidth=1)

axes[1].plot(t, sol.y[1, :], linewidth=5)
axes[1].plot(t, mysol[1, :])

axes[2].plot(t, sol.y[2, :], linewidth=5)
axes[2].plot(t, mysol[2, :], linewidth=1)

plt.show()
