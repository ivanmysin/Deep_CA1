import numpy as np
import matplotlib.pyplot as plt

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