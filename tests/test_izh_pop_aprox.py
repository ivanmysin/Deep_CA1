import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import izhs_lib

from scipy.signal.windows import parzen


win  = parzen(51)
win = win / np.sum(win)

def mf_izh_ode(t, y, constants):
    """
    Функция, возвращающая правые части системы дифференциальных уравнений.

    :param y: массив текущих значений переменных [r, <v>, <w>].
    :param t: текущее время.
    :param constants: словарь с постоянными величинами.
    :return: список производных dr/dt, d<v>/dt, d<w>/dt
    """
    r, v_avg, w_avg = y # , s
    Delta_eta = constants['Delta_eta']
    alpha = constants['alpha']
    bar_eta = constants['bar_eta']
    I_ext = constants['I_ext']

    e_r = 1.0 #constants['e_r']
    a = constants['a']
    b = constants['b']
    w_jump = constants['w_jump']
    tau_pop = constants['tau_pop']


    g_syn = constants['gsyn_max']


    dr_dt = (Delta_eta / np.pi + 2 * r * v_avg - (alpha + g_syn) * r) / tau_pop
    dv_avg_dt = (v_avg ** 2 - alpha * v_avg - w_avg + bar_eta + I_ext + g_syn * (e_r - v_avg) - np.pi**2 * r**2) / tau_pop
    dw_avg_dt = (a * (b * v_avg - w_avg) + w_jump * r) / tau_pop


    return [dr_dt, dv_avg_dt, dw_avg_dt] # , ds_dt
##############################################################
dim_izh_params = {
    "Cm": 477.0, # 114, # * pF,  # /cm**2,
    "k": 2.47616920162969, # 1.19, # * mS / mV,
    "Vrest": -63.09516295101267, #-57.63, # * mV,
    "Vth": -51.933812933530746, #-35.53, #*mV, # np.random.normal(loc=-35.53, scale=4.0, size=NN) * mV,  # -35.53*mV,
    "Vpeak": 27.11825425457242, # * mV,
    "Vreset": -56.14747086191443, # * mV,
    "a": 0.016342557423710602, # 0.005, # * ms ** -1,
    "b": -12.195992950332169, #0.22, # * mS,
    "d": 460.0, # 2, # * pA,

    "Iext" : 0.0,

    "gsyn_max": 20,
}


# Словарь с константами
cauchy_dencity_params = {
    'Delta_eta': 0.9 * dim_izh_params['Cm'], # 0.02,
    'bar_eta': 0, # 0.191,
}

dim_izh_params = dim_izh_params | cauchy_dencity_params

izh_params = izhs_lib.dimensional_to_dimensionless_all(dim_izh_params)
# izh_params['v_peak'] = 1.2
# izh_params['v_reset'] = -0.5

print(izh_params['gsyn_max'])



# Временной интервал
duration = 200
dt = 0.01
t = np.linspace(0, duration, int(duration/dt))

# Решение системы ОДУ
# Начальные условия
y0 = np.asarray([0.0, 0.0, 0.0]) # , 0
solution = solve_ivp(mf_izh_ode, t_span=[0, duration], y0=y0, t_eval=t, args=(izh_params,), method='RK45')


# direct_r, direct_v_avg, direct_u_avg = izhs_lib.izh_nondim_simulate(izh_params, izh_params, dt=dt, duration=duration, NN=1)


direct_r2, direct_v_avg2, direct_u_avg2 = izhs_lib.izh_simulate(dim_izh_params, cauchy_dencity_params, dt=dt, duration=duration, NN=1000)
direct_v_avg2 = 1 + direct_v_avg2 / (np.abs(dim_izh_params['Vrest']))
direct_u_avg2 = direct_u_avg2 / (dim_izh_params['k'] * dim_izh_params['Vrest']**2)

direct_r = np.convolve(direct_r2, win, mode='same')

# r_equlib = 1000 * izh_params['Delta_eta'] / np.pi  / izh_params['alpha']

# Извлекаем результаты
r = solution.y[0, :] * 1000  # / dim_izh_params['Cm'] * dim_izh_params['k'] * np.abs(dim_izh_params['Vrest'])
v_avg = solution.y[1, :]
w_avg = solution.y[2, :]
#s = solution[:, 3]

fig, axes = plt.subplots(nrows=3, sharex=True)
axes[0].plot(t, r, linewidth=3)
axes[0].plot(t, direct_r)

# axes[0].hlines([r_equlib, r_equlib], xmin=0, xmax=t[-1], color='red')


axes[1].plot(t, v_avg, linewidth=3)
axes[1].plot(t, direct_v_avg2)
#axes[1].plot(t, direct_v_avg)


axes[1].set_ylim(-1, 2)

axes[2].plot(t, w_avg, linewidth=3)
# axes[2].plot(t, direct_u_avg)
axes[2].plot(t, direct_u_avg2)



#axes[3].plot(t, s)

plt.show()

