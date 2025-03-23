import numpy as np
import matplotlib.pyplot as plt
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

def izh_simulate(params, eta_params, dt=0.1, duration=200, NN=4000):
    k = params['k']
    Vrest =  params['Vrest']
    VT =  params['Vth']
    Iext =  params['Iext']
    Cm =  params['Cm']
    a =  params['a']
    b =  params['b']
    Vpeak = params['Vpeak']
    Vreset = params['Vmin']
    d =  params['d']

    V = np.zeros(NN, dtype=float) + params['V0']
    U = np.zeros(NN, dtype=float) + params['U0']

    eta = eta_params['bar_eta'] + eta_params['Delta_eta'] * np.tan(np.pi*(np.random.rand(NN) - 0.5) )
    #eta = 1000.1 * eta #* k * Vreset**2
    print(np.sort(eta))
    # plt.hist(eta, bins=100)
    # plt.show()

    Nt = int(duration / dt)

    firings = np.zeros(Nt, dtype=float)
    v_avg = np.zeros(Nt, dtype=float)
    u_avg = np.zeros(Nt, dtype=float)

    for ts_idx in range(Nt):

        dVdt = (k * (V - Vrest) * (V - VT) - U + eta + Iext) / Cm
        dUdt = a * (b * (V - Vrest) - U)

        V = V + dt * dVdt
        U = U + dt * dUdt

        fired = V > Vpeak
        V[fired] = Vreset
        U[fired] += d

        firings[ts_idx] = np.mean(fired)
        v_avg[ts_idx] = np.mean(V)
        u_avg[ts_idx] = np.mean(U)

    return firings


def izh_nondim_simulate(params, eta_params, dt=0.1, duration=200, NN=4000):

    alpha = params['alpha']

    a =  params['a']
    b =  params['b']
    Vpeak = params['v_peak']
    Vreset = params['v_reset']
    d =  params['w_jump']
    Iext =  params['I_ext']

    print(Vreset, Vpeak)

    V = np.zeros(NN, dtype=float) + params['vk']
    U = np.zeros(NN, dtype=float) + params['wk']

    eta = eta_params['bar_eta'] + eta_params['Delta_eta'] * np.random.standard_cauchy(NN)  #np.tan(np.pi*(np.random.rand(NN) - 0.5) )
    eta = np.sort(eta)
    print(eta)

    Nt = int(duration / dt)

    firings = np.zeros(Nt, dtype=float)
    v_avg = np.zeros(Nt, dtype=float)
    u_avg = np.zeros(Nt, dtype=float)

    for ts_idx in range(Nt):

        dVdt = V * (V - alpha) - U + eta + Iext
        dUdt = a * (b*V - U)

        #print(dt * dVdt[-1])

        V = V + dt * dVdt
        U = U + dt * dUdt

        fired = V > Vpeak
        V[fired] = Vreset
        U[fired] += d

        firings[ts_idx] = np.mean(fired) / dt
        v_avg[ts_idx] = np.mean(V)
        u_avg[ts_idx] = np.mean(U)

    return firings, v_avg, u_avg