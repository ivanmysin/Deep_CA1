import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import izhs_lib

def mf_izh_ode(t, y, constants):
    """
    Функция, возвращающая правые части системы дифференциальных уравнений.

    :param y: массив текущих значений переменных [r, <v>, <w>, s].
    :param t: текущее время.
    :param constants: словарь с постоянными величинами.
    :return: список производных dr/dt, d<v>/dt, d<w>/dt, ds/dt.
    """
    r, v_avg, w_avg = y # , s
    Delta_eta = constants['Delta_eta']
    alpha = constants['alpha']
    bar_eta = constants['bar_eta']
    I_ext = constants['I_ext']
    g_syn = 0.0 # 0.002 #constants['g_syn']
    e_r = 1.0 #constants['e_r']
    a = constants['a']
    b = constants['b']
    w_jump = constants['w_jump']
    # tau_s = constants['tau_s']
    # s_jump = constants['s_jump']
    s = ( 0.5*np.cos(2*np.pi*8*0.001*t) + 0.5)**4

    #I_ext += 50 * ( 0.5*np.cos(2*np.pi*8*0.001*t) + 0.5)**4

    dr_dt = Delta_eta / np.pi + 2 * r * v_avg - (alpha + g_syn * s) * r
    dv_avg_dt = v_avg ** 2 - alpha * v_avg - w_avg + bar_eta + I_ext + g_syn * s * (e_r - v_avg) - np.pi**2 * r**2
    dw_avg_dt = a * (b * v_avg - w_avg) + w_jump * r
    #ds_dt = -s / tau_s + s_jump * r

    return [dr_dt, dv_avg_dt, dw_avg_dt] # , ds_dt
##############################################################
dim_izh_params = {
    "V0" : -60.0,
    "U0" : 0.0,

    "Cm": 38.0, #114, # * pF,  # /cm**2,
    "k":  0.16, # # 1.19, # * mS / mV,
    "Vrest": -60.93990874, # -57.63, # * mV,
    "Vth": -46.188382, #-35.53, #*mV, # np.random.normal(loc=-35.53, scale=4.0, size=NN) * mV,  # -35.53*mV,
    "Vpeak": -0.91960626, # 21.72, # * mV,
    "Vmin": -55.33551486, # -48.7, # * mV,
    "a": 0.00391281, # 0.005, # * ms ** -1,
    "b": 2.72086524, #0.22, # * mS,
    "d": 3.0, #2, # * pA,

    "Iext" : 0.0,
}


# Словарь с константами
cauchy_dencity_params = {
    'Delta_eta': 2, # 0.02,
    'bar_eta': 0.0, # 0.191,
}

dim_izh_params = dim_izh_params | cauchy_dencity_params

izh_params = izhs_lib.dimensional_to_dimensionless(dim_izh_params)
izh_params['v_peak'] = 200
izh_params['v_reset'] = -200

# Начальные условия
y0 = np.asarray([0.0, 2.2, 0.0]) # , 0

# Временной интервал
duration = 500
dt = 0.05
t = np.linspace(0, duration, int(duration/dt))

# Решение системы ОДУ
solution = solve_ivp(mf_izh_ode, t_span=[0, duration], y0=y0, t_eval=t, args=(izh_params,), method='RK23')

# direct_r, direct_v_avg, direct_u_avg = izhs_lib.izh_nondim_simulate(izh_params, izh_params, dt=dt, duration=duration, NN=10000)
#direct_r, direct_v_avg, direct_u_avg = izhs_lib.izh_simulate(izh_params, cauchy_dencity_params, dt=dt, duration=duration, NN=10000)




r_equlib = 1000 * izh_params['Delta_eta'] / np.pi  / izh_params['alpha']

# Извлекаем результаты
r = solution.y[0, :] * 1000
v_avg = solution.y[1, :]
w_avg = solution.y[2, :]
#s = solution[:, 3]

fig, axes = plt.subplots(nrows=3, sharex=True)
#axes[0].plot(t, direct_r)
axes[0].plot(t, r, linewidth=3)
axes[0].hlines([r_equlib, r_equlib], xmin=0, xmax=t[-1], color='red')

#axes[1].plot(t, direct_v_avg)
axes[1].plot(t, v_avg, linewidth=3)
axes[1].set_ylim(-1, 2)

#axes[2].plot(t, direct_u_avg)
axes[2].plot(t, w_avg, linewidth=3)

#axes[3].plot(t, s)

plt.show()

