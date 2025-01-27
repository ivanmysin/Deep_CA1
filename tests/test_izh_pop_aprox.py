import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import izhs_lib

def transform_vk(V_k, V_R):
    return 1 + V_k / abs(V_R)

def transform_wk(W_k, k1, V_R):
    return W_k / (k1 * abs(V_R)**2)

def transform_s(s):
    return s

def transform_T(t, C, k1, V_R):
    return C * t / (k1 * abs(V_R))

def transform_v_peak(V_peak, V_R):
    return 1 + V_peak / abs(V_R)

def transform_v_reset(V_reset, V_R):
    return 1 + V_reset / abs(V_R)

def transform_alpha(V_T, V_R):
    return 1 + V_T / abs(V_R)

def transform_g_syn(G_syn, k1, V_R):
    return G_syn / (k1 * abs(V_R))

def transform_a(tau_W, k1, V_R, C):
    return 1 / ((tau_W * k1 * abs(V_R)) / C)

def transform_b(beta, k1, V_R):
    return beta / (k1 * abs(V_R))

def transform_s_jump(S_jump, C, k1, V_R):
    return S_jump * C / (k1 * abs(V_R))

def transform_w_jump(W_jump, k1, V_R):
    return W_jump / (k1 * abs(V_R)**2)

def transform_e_r(E_r, V_R):
    return 1 + E_r / abs(V_R)

def transform_I(I_app, k1, V_R):
    return I_app / (k1 * abs(V_R)**2)

def transform_tau_s(tau_syn, k1, V_R, C):
    return tau_syn * k1 * abs(V_R) / C


def dimensional_to_dimensionless(dimensional_vars):
    k1 = dimensional_vars['k']
    C  = dimensional_vars['Cm']
    V_R = dimensional_vars['Vrest']


    dimensionless_vars = {}

    # Преобразования
    dimensionless_vars['vk'] = transform_vk(dimensional_vars['V0'], V_R)
    dimensionless_vars['wk'] = transform_wk(dimensional_vars['U0'], k1, V_R)
    #dimensionless_vars['s'] = transform_s(dimensional_vars['s'])
    #dimensionless_vars['T'] = transform_T(dimensional_vars['t'], C, k1, V_R)
    dimensionless_vars['v_peak'] = transform_v_peak(dimensional_vars['Vpeak'], V_R)
    dimensionless_vars['v_reset'] = transform_v_reset(dimensional_vars['Vmin'], V_R)
    dimensionless_vars['alpha'] = transform_alpha(dimensional_vars['Vth'], V_R)
    #dimensionless_vars['g_syn'] = transform_g_syn(dimensional_vars['G_syn'], k1, V_R)
    dimensionless_vars['a'] = transform_a(1/dimensional_vars['a'], k1, V_R, C)
    dimensionless_vars['b'] = transform_b(dimensional_vars['b'], k1, V_R)
    #dimensionless_vars['s_jump'] = transform_s_jump(dimensional_vars['S_jump'], C, k1, V_R)
    dimensionless_vars['w_jump'] = transform_w_jump(dimensional_vars['d'], k1, V_R)

    #dimensionless_vars['e_r'] = transform_e_r(dimensional_vars['E_r'], V_R)

    dimensionless_vars['I_ext'] = transform_I(dimensional_vars['Iext'], k1, V_R)
    #dimensionless_vars['tau_s'] = transform_tau_s(dimensional_vars['tau_syn'], k1, V_R, C)

    return dimensionless_vars

def mf_izh_ode(y, t, constants):
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
    g_syn = 0.0 #constants['g_syn']
    e_r = 0.0 #constants['e_r']
    a = constants['a']
    b = constants['b']
    w_jump = constants['w_jump']
    # tau_s = constants['tau_s']
    # s_jump = constants['s_jump']
    s = 0

    dr_dt = Delta_eta / np.pi + 2 * r * v_avg - (alpha + g_syn * s) * r
    dv_avg_dt = v_avg ** 2 - alpha * v_avg - w_avg + bar_eta + I_ext + g_syn * s * (e_r - v_avg) - np.pi**2 * r**2
    dw_avg_dt = a * (b * v_avg - w_avg) + w_jump * r
    #ds_dt = -s / tau_s + s_jump * r

    return [dr_dt, dv_avg_dt, dw_avg_dt] # , ds_dt
##############################################################
dim_izh_params = {
    "V0" : -57.0,
    "U0" : 0.0,

    "Cm": 114, # * pF,  # /cm**2,
    "k": 1.19, # * mS / mV,
    "Vrest": -57.63, # * mV,
    "Vth": -35.53, #*mV, # np.random.normal(loc=-35.53, scale=4.0, size=NN) * mV,  # -35.53*mV,
    "Vpeak": 21.72, # * mV,
    "Vmin": -48.7, # * mV,
    "a": 0.005, # * ms ** -1,
    "b": 0.22, # * mS,
    "d": 2, # * uA,

    "Iext" : 580,
}


# Словарь с константами
cauchy_dencity_params = {
    'Delta_eta': 0.02,
    'bar_eta': 0.0, # 0.191,
}

# izh_params = {
#     'alpha': 0.6215,
#     'I_ext': 0.2,
#     'g_syn': 1.2308,
#     'e_r': 1,
#     'a': 0.0077,
#     'b': -0.0062,
#     'w_jump':  0.0189,
#     'tau_s': 2.6,
#     's_jump': 1.2308,
#     'v_peak' : 200,
#     'v_reset' : -200,
#     'vk' : 0,
#     'wk' : 0,
# }

izh_params = dimensional_to_dimensionless(dim_izh_params)
#izh_params['v_peak'] = 200
#izh_params['v_reset'] = -200


params = cauchy_dencity_params | izh_params

# Начальные условия
y0 = [0.0, izh_params['vk'], izh_params['wk']] # , 0

# Временной интервал
duration = 50
dt = 0.001
t = np.linspace(0, duration, int(duration/dt))

# Решение системы ОДУ
solution = odeint(mf_izh_ode, y0, t, args=(params,))

direct_r, direct_v_avg, direct_u_avg = izhs_lib.izh_nondim_simulate(izh_params, cauchy_dencity_params, dt=dt, duration=duration, NN=10000)
#direct_r, direct_v_avg, direct_u_avg = izhs_lib.izh_simulate(izh_params, cauchy_dencity_params, dt=dt, duration=duration, NN=10000)

# Извлекаем результаты
r = solution[:, 0]
v_avg = solution[:, 1]
w_avg = solution[:, 2]
#s = solution[:, 3]

fig, axes = plt.subplots(nrows=3)
axes[0].plot(t, direct_r)
axes[0].plot(t, r, linewidth=3)

axes[1].plot(t, direct_v_avg)
axes[1].plot(t, v_avg, linewidth=3)

axes[2].plot(t, direct_u_avg)
axes[2].plot(t, w_avg, linewidth=3)

#axes[3].plot(t, s)

plt.show()

