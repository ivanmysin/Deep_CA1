import numpy as np
import matplotlib.pyplot as plt
import izhs_lib
import pandas as pd
import os
os.chdir("../")
import myconfig
SimulatedType = "CA1 Perforant Path-Associated"  #  "CA1 Basket" #


def w_avg_solution(t, a, b, v_avg, r, w_jump, w0):
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
    # Вычисляем стационарную часть: b*v_avg + (r*w_jump)/a
    stationary_term_1 = b * v_avg  # (N,)

    # Защита от деления на ноль для a
    safe_a = np.where(np.abs(a) < 1e-10, np.ones_like(a), a)  # (N,)
    stationary_term_2 = (r * w_jump) / safe_a  # (N,)

    stationary = stationary_term_1 + stationary_term_2  # (N,)
    #stationary = tf.expand_dims(stationary, axis=1)  # (N, 1)

    # C1 = w0 - stationary
    C1 = w0 - stationary  # (N, 1)

    # Экспоненциальный множитель: exp(-a*t)
    # a_expanded = tf.expand_dims(a, axis=1)  # (N, 1)
    exp_term = np.exp(-a * t)  # (N, T)

    # Итоговое решение
    w_vals = C1 * exp_term + stationary  # (N, T)

    return w_vals

neurons_params = pd.read_csv(myconfig.IZHIKEVICNNEURONSPARAMS)
neurons_params = neurons_params.fillna(-1)
neurons_params = neurons_params.rename(columns={'Izh Vr': 'Vrest', 'Izh Vt': 'Vth', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd', 'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vreset',})
neurons_params = neurons_params.drop(['CARLsim_default', 'E/I', 'Population Size', 'Refractory Period', 'rank'], axis=1)
tp = neurons_params[ neurons_params["Neuron Type"] == SimulatedType].iloc[0]


dim_izh_params = tp.to_dict()

print(dim_izh_params)

dim_izh_params['V0'] = dim_izh_params['Vrest']
dim_izh_params['U0'] = 0.0
dim_izh_params['Iext'] = 200

NN = 2
dt_dim = 0.01 # ms
duration = 400

# dim_izh_params = {
#     "V0" : -57.63,
#     "U0" : 0.0,
#
#     "Cm": 114, # * pF,
#     "k": 1.19, # * mS
#     "Vrest": -57.63, # * mV,
#     "Vth": -35.53, #*mV, # np.random.normal(loc=-35.53, scale=4.0, size=NN) * mV,  # -35.53*mV,
#     "Vpeak": 21.72, # * mV,
#     "Vreset": -48.7, # * mV,
#     "a": 0.005, # * ms ** -1,
#     "b": 0.22, # * mS,
#     "d": 2, # * pA,
#
#     "Iext" : 200, # pA
# }

# Словарь с константами
cauchy_dencity_params = {
    'Delta_eta': 5,  #0.02,
    'bar_eta': 0.0, # 0.191,
}

dim_izh_params = dim_izh_params | cauchy_dencity_params

izh_params = izhs_lib.dimensional_to_dimensionless(dim_izh_params)
izh_params['v_peak'] = 2
izh_params['v_reset'] = -2

params = cauchy_dencity_params | izh_params

dt_non_dim = izhs_lib.transform_T(dt_dim, dim_izh_params['Cm'], dim_izh_params['k'], dim_izh_params['Vrest'])

Delta_eta = params['Delta_eta']
alpha = params['alpha']
bar_eta = params['bar_eta']
I_ext = params['I_ext']
w_jump = params['w_jump']
a = params['a']
b = params['b']


rate = np.zeros(NN, dtype=np.float64)
v_avg = np.zeros_like(rate) # 1.0 # izh_params['vk']
#v_avg[0] = -0.05

w_avg = np.zeros_like(rate) + izh_params['wk']

## synaptic static variables
tau_d = 6.02  # ms
tau_r = 359.8 # ms
tau_f = 21.0  # ms
Uinc = 0.25

if  tau_d != tau_r:
    tau1r = tau_d / (tau_d - tau_r)
else:
    tau1r = 1e-13

gsyn_max = np.zeros(shape=(NN, NN), dtype=np.float64)
# gsyn_max[0, 1] = 20
# gsyn_max[1, 0] = 15

Erev = np.zeros(shape=(NN, NN), dtype=np.float64) - 75
e_r = izhs_lib.transform_e_r(Erev, dim_izh_params['Vrest'])

## synaptic dynamic variables
A = np.zeros(shape=(NN, NN), dtype=np.float64)
R = np.ones_like(A)
U = np.zeros_like(A)


rates = []
t = np.arange(0, duration, dt_dim)

print('alpha =', alpha)
print('a =', a)
print('b =', b)
print('Delta_eta =', Delta_eta)
print('dt_non_dim =', dt_non_dim)
print('w_jump =', w_jump)
print('I_ext =', I_ext)


for ts in t:
    g_syn = gsyn_max * A

    g_syn_tot = np.sum(g_syn, axis=0)

    Isyn = np.sum( g_syn * (e_r - v_avg.reshape(1, -1)), axis=0 )

    rate[0] = rate[0] + dt_non_dim * (Delta_eta / np.pi + 2 * rate[0] * v_avg[0] - (alpha + g_syn_tot[0]) * rate[0])

    exponent_base = -alpha - g_syn_tot[1] + 2.0 * v_avg[1]
    exponent = exponent_base * dt_non_dim  #
    exp_term = np.exp(exponent)  # (N, T)
    denominator = alpha + g_syn_tot[1] - 2.0 * v_avg[1]
    stationary = Delta_eta / (np.pi * denominator)  # (N, 1)
    C1 = rate[1] - stationary  # (N, 1)
    # Итоговое решение: C1 * exp_term + stationary
    rate[1] = C1 * exp_term + stationary  # (N, T)


    v_avg = v_avg + dt_non_dim * (v_avg ** 2 - alpha * v_avg - w_avg + bar_eta + I_ext + Isyn - (np.pi * rate)**2)


    w_avg[0] = w_avg[0] + dt_non_dim * (a * (b * v_avg[0] - w_avg[0]) + w_jump * rate[0])

    w_avg[1] = w_avg_solution(dt_non_dim, a, b, v_avg[1], rate[1], w_jump, w_avg[1])

    firing_prob = dt_non_dim * rate
    firing_prob = firing_prob.reshape(-1, 1)


    a_ = A * np.exp(-dt_dim / tau_d)
    r_ = 1 + (R - 1 + tau1r * A) * np.exp(-dt_dim / tau_r) - tau1r * A
    u_ = U * np.exp(-dt_dim / tau_f)

    U = u_ + Uinc * (1 - u_) * firing_prob
    A = a_ + U * r_ * firing_prob
    R = r_ - U * r_ * firing_prob

    rates.append( np.copy(rate) )

rates = np.stack(rates) * dt_non_dim / dt_dim * 1000


plt.plot(t, rates)
plt.show()

