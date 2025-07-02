import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0 as bessel_i0

import izhs_lib
# import h5py
#
import sys
sys.path.append('/')
import myconfig

class MeanFieldNetwork:

    def __init__(self, params, dt_dim=0.01, use_input=False, **kwargs):

        self.v_threshold = 2.0
        self.dt_dim = dt_dim
        self.use_input = use_input

        self.units = len(params['alpha'])
        self.alpha = np.asarray( params['alpha'], dtype=myconfig.DTYPE )
        self.a = np.asarray( params['a'], dtype=myconfig.DTYPE )
        self.b = np.asarray( params['b'], dtype=myconfig.DTYPE )
        self.w_jump = np.asarray( params['w_jump'], dtype=myconfig.DTYPE )
        self.dts_non_dim = np.asarray( params['dts_non_dim'], dtype=myconfig.DTYPE )
        self.Delta_eta = np.asarray( params['Delta_eta'], dtype=myconfig.DTYPE)

        self.I_ext = np.asarray( params['I_ext'], dtype=myconfig.DTYPE )

        self.gsyn_max = np.asarray(params['gsyn_max'], dtype=myconfig.DTYPE)
        self.tau_f = np.asarray(params['tau_f'], dtype=myconfig.DTYPE)
        self.tau_d = np.asarray(params['tau_d'], dtype=myconfig.DTYPE)
        self.tau_r = np.asarray(params['tau_r'], dtype=myconfig.DTYPE)
        self.Uinc = np.asarray(params['Uinc'], dtype=myconfig.DTYPE)

        self.e_r = np.asarray(params['e_r'], dtype=myconfig.DTYPE)
        self.pconn = np.asarray(params['pconn'])


        synaptic_matrix_shapes = self.gsyn_max.shape

        self.tau1r = np.where(self.tau_d != self.tau_r, self.tau_d / (self.tau_d - self.tau_r), 1e-13)

        self.exp_tau_d = np.exp(-self.dt_dim / self.tau_d)
        self.exp_tau_f = np.exp(-self.dt_dim / self.tau_f)
        self.exp_tau_r = np.exp(-self.dt_dim / self.tau_r)

        self.state_size = [self.units, self.units, self.units, synaptic_matrix_shapes, synaptic_matrix_shapes, synaptic_matrix_shapes]

    def get_initial_state(self, batch_size=1):
        shape = [self.units+3, self.units]

        r = np.zeros( [1, self.units], dtype=myconfig.DTYPE)
        v = np.zeros( [1, self.units], dtype=myconfig.DTYPE)
        w = np.zeros( [1, self.units], dtype=myconfig.DTYPE)

        synaptic_matrix_shapes = self.gsyn_max.shape

        R = np.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)
        U = np.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)
        A = np.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)

        initial_state = [r, v, w, R, U, A]

        return initial_state

    def get_rate_derivative(self, rates, v_avg, g_syn_tot):
        drdt = self.Delta_eta / np.pi + 2 * rates * v_avg - (self.alpha + g_syn_tot) * rates
        return drdt

    def get_v_avg_derivative(self, rates, v_avg, w_avg, g_syn):
        Isyn = np.sum(g_syn * (self.e_r - v_avg), axis=0)
        dvdt = v_avg ** 2 - self.alpha * v_avg - w_avg + self.I_ext + Isyn - (np.pi * rates)**2
        return dvdt

    def get_w_avg_derivative(self, rates, v_avg, w_avg):
        dwdt = self.a * (self.b * v_avg - w_avg) + self.w_jump * rates
        return dwdt

    def runge_kutta_step(self, rates, v_avg, w_avg, g_syn):
        g_syn_tot = np.sum(g_syn, axis=0)

        k1_rates = self.get_rate_derivative(rates, v_avg, g_syn_tot)
        k1_v = self.get_v_avg_derivative(rates, v_avg, w_avg, g_syn)
        k1_w = self.get_w_avg_derivative(rates, v_avg, w_avg)

        half_rate = rates + 0.5 * self.dts_non_dim * k1_rates
        half_v = v_avg + 0.5 * self.dts_non_dim * k1_v
        half_w = w_avg + 0.5 * self.dts_non_dim * k1_w


        k2_rates = self.get_rate_derivative(half_rate, half_v, g_syn_tot)
        k2_v = self.get_v_avg_derivative(half_rate, half_v, half_w, g_syn)
        k2_w = self.get_w_avg_derivative(half_rate, half_v, half_w)

        half_rate = rates + 0.5 * self.dts_non_dim * k2_rates
        half_v = v_avg + 0.5 * self.dts_non_dim *  k2_v
        half_w = w_avg + 0.5 * self.dts_non_dim * k2_w

        k3_rates = self.get_rate_derivative(half_rate, half_v, g_syn_tot)
        k3_v = self.get_v_avg_derivative(half_rate, half_v, half_w, g_syn)
        k3_w = self.get_w_avg_derivative(half_rate, half_v, half_w)

        half_rate = rates + self.dts_non_dim * k3_rates
        half_v = v_avg + self.dts_non_dim * k3_v
        half_w = w_avg + self.dts_non_dim * k3_w

        k4_rates = self.get_rate_derivative(half_rate, half_v, g_syn_tot)
        k4_v = self.get_v_avg_derivative(half_rate, half_v, half_w, g_syn)
        k4_w = self.get_w_avg_derivative(half_rate, half_v, half_w)

        rates = rates + self.dts_non_dim * (k1_rates + 2*k2_rates + 2*k3_rates + k4_rates) / 6.0
        v_avg = v_avg + self.dts_non_dim * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0
        w_avg = w_avg + self.dts_non_dim * (k1_w + 2*k2_w + 2*k3_w + k4_w) / 6.0

        return rates, v_avg, w_avg

    def call(self, inputs, states):
        rates = states[0]
        v_avg = states[1]
        w_avg = states[2]
        R = states[3]
        U = states[4]
        A = states[5]

        g_syn = self.gsyn_max * A
        # g_syn_tot = np.sum(g_syn, axis=0)
        #
        #
        # Isyn = np.sum(g_syn * (self.e_r - v_avg), axis=0)
        #
        # drates = self.dts_non_dim * (self.Delta_eta / np.pi + 2 * rates * v_avg - (self.alpha + g_syn_tot) * rates)
        # dv_avg = self.dts_non_dim * (v_avg**2 - self.alpha * v_avg - w_avg + self.I_ext + Isyn - (np.pi *rates)**2)
        # dw_avg = self.dts_non_dim * (self.a * (self.b * v_avg - w_avg) + self.w_jump * rates)
        #
        # rates = rates + drates
        # v_avg = v_avg + dv_avg
        # w_avg = w_avg + dw_avg

        rates, v_avg, w_avg = self.runge_kutta_step(rates, v_avg, w_avg, g_syn)

        firing_probs =  (self.dts_non_dim * rates).T #tf.reshape(rates, shape=(-1, 1))

        if self.use_input:
            inputs = inputs.T * 0.001 * self.dt_dim
            firing_probs = np.concatenate( [firing_probs, inputs], axis=0)

        v_avg[ v_avg > self.v_threshold ] = self.v_threshold
        #v_avg[v_avg < -0.5] = -1

        FRpre_normed = self.pconn *  firing_probs




        a_ = A * self.exp_tau_d
        r_ = 1 + (R - 1 + self.tau1r * A) * self.exp_tau_r  - self.tau1r * A
        u_ = U * self.exp_tau_f

        U = u_ + self.Uinc * (1 - u_) * FRpre_normed
        A = a_ + U * r_ * FRpre_normed
        R = r_ - U * r_ * FRpre_normed


        output = rates * self.dts_non_dim / self.dt_dim * 1000 # convert to spike per second

        return output, [rates, v_avg, w_avg, R, U, A]

    def predict(self, inputs, time_axis=1, initial_states=None):
        if initial_states is None:
            states = self.get_initial_state()
        else:
            states = initial_states
        outputs = []
        hist_states = []
        for idx in range(inputs.shape[time_axis]):
            inp = inputs[:, idx, :]

            output, states = self.call(inp, states)

            outputs.append(output)

            for s in states:
                hist_states.append(s)

        outputs = np.stack(outputs)

        #hist_states = np.stack(hist_states)
        # print('outputs', outputs.shape)
        h_states = []
        for s_idx in range(len(states)):
            s = hist_states[s_idx::len(states)]
            #print(s[0].shape, s[-1].shape)

            s = np.stack(s)

            h_states.append(s)
        return outputs, h_states

class SpatialThetaGenerators:
    def __init__(self, params, **kwargs):

        self.ALPHA = 5.0

        if type(params) == list:

            ThetaFreq = []

            OutPlaceFiringRate = []
            OutPlaceThetaPhase = []
            InPlacePeakRate = []
            CenterPlaceField = []
            Rs = []
            SigmaPlaceField = []
            SlopePhasePrecession = []
            PrecessionOnset = []

            self.n_outs = len(params)


            for p in params:
                OutPlaceFiringRate.append( p["OutPlaceFiringRate"] )
                OutPlaceThetaPhase.append( p["OutPlaceThetaPhase"] )
                InPlacePeakRate.append( p["InPlacePeakRate"] )
                Rs.append(p["R"]),

                CenterPlaceField.append(p["CenterPlaceField"])

                SigmaPlaceField.append(p["SigmaPlaceField"])
                SlopePhasePrecession.append(p["SlopePhasePrecession"])
                PrecessionOnset.append(p["PrecessionOnset"])


                ThetaFreq.append( p["ThetaFreq"] )
        else:
            ThetaFreq = params['ThetaFreq']

            OutPlaceFiringRate = params['OutPlaceFiringRate']
            OutPlaceThetaPhase = params['OutPlaceThetaPhase']
            InPlacePeakRate = params['InPlacePeakRate']
            CenterPlaceField = params['CenterPlaceField']
            Rs = params['R']
            SigmaPlaceField = params['SigmaPlaceField']
            SlopePhasePrecession = params['SlopePhasePrecession']
            PrecessionOnset = params['PrecessionOnset']

            self.n_outs = len(OutPlaceFiringRate)

        self.ThetaFreq = np.asarray(ThetaFreq, dtype=myconfig.DTYPE).reshape([1, -1])
        self.OutPlaceFiringRate = np.asarray(OutPlaceFiringRate, dtype=myconfig.DTYPE).reshape([1, -1])
        self.OutPlaceThetaPhase = np.asarray(OutPlaceThetaPhase, dtype=myconfig.DTYPE).reshape([1, -1])
        self.R = np.asarray(Rs, dtype=myconfig.DTYPE).reshape( [1, -1])
        self.InPlacePeakRate = np.asarray(InPlacePeakRate, dtype=myconfig.DTYPE).reshape([1, -1])
        self.CenterPlaceField = np.asarray(CenterPlaceField, dtype=myconfig.DTYPE).reshape([1, -1])
        self.SigmaPlaceField = np.asarray(SigmaPlaceField, dtype=myconfig.DTYPE).reshape([1, -1])
        self.SlopePhasePrecession = np.asarray(SlopePhasePrecession, dtype=myconfig.DTYPE).reshape([1, -1])
        self.PrecessionOnset = np.asarray(PrecessionOnset, dtype=myconfig.DTYPE).reshape([1, -1])

        Mask = np.isnan(self.PrecessionOnset)
        self.PrecessionOnset = np.where(Mask, 0., self.PrecessionOnset )
        self.PhaseOnsetMask = Mask.astype(dtype=myconfig.DTYPE) - 1.0

        self.kappa = self.r2kappa(self.R)

        self.mult4time = 2 * np.pi * self.ThetaFreq * 0.001  # tf.constant(tmp, dtype=myconfig.DTYPE)

        I0 = bessel_i0(self.kappa)
        self.normalizator = self.OutPlaceFiringRate / I0

    def r2kappa(self, R):
        """
        Recalculate kappa from R for Von Mises distribution using vectorized operations with NumPy arrays.

        Parameters:
        -----------
        R : array-like
            Input values of correlation coefficient or concentration parameter.

        Returns:
        --------
        kappa : ndarray
            Transformed kappa values corresponding to input R values.
        """
        # Ensure that the input is a NumPy array
        R = np.asarray(R)

        # Create masks based on conditions
        mask1 = R < 0.53
        mask2 = (R >= 0.53) & (R < 0.85)
        mask3 = R >= 0.85

        # Initialize an empty result array
        kappa = np.empty_like(R)

        # Apply formulas conditionally
        kappa[mask1] = 2 * R[mask1] + R[mask1] ** 3 + 5 / 6 * R[mask1] ** 5
        kappa[mask2] = -0.4 + 1.39 * R[mask2] + 0.43 / (1 - R[mask2])
        kappa[mask3] = 1 / (3 * R[mask3] - 4 * R[mask3] ** 2 + R[mask3] ** 3)

        return kappa


    def call(self, t):
        ampl4gauss = 2 * (self.InPlacePeakRate - self.OutPlaceFiringRate) / (self.OutPlaceFiringRate + 1)
        multip = (1 + ampl4gauss * np.exp(-0.5 * ((t - self.CenterPlaceField) / self.SigmaPlaceField) ** 2))

        start_place = t - self.CenterPlaceField - 3 * self.SigmaPlaceField
        end_place = t - self.CenterPlaceField + 3 * self.SigmaPlaceField



        inplace = 0.25 * (1.0 - (start_place / (self.ALPHA + np.abs(start_place)))) * (
                1.0 + end_place / (self.ALPHA + abs(end_place)))

        precession = self.SlopePhasePrecession * inplace
        phases = self.OutPlaceThetaPhase * (1 - inplace * self.PhaseOnsetMask ) + self.PrecessionOnset * inplace * self.PhaseOnsetMask


        firings = self.normalizator * np.exp(self.kappa * np.cos((self.mult4time + precession) * t - phases))

        firings = multip * firings  # / (0.001 * dt)
        firings = firings.reshape( [1, firings.shape[1], firings.shape[2]] )

        return firings



######################################################################
if __name__ == '__main__':

    NN = 2
    Ninps = 3
    dt_dim = 0.1  # ms
    duration = 1000.0

    dim_izh_params = {
        "V0": -57.63,
        "U0": 0.0,

        "Cm": 114,  # * pF,
        "k": 1.19,  # * mS
        "Vrest": -57.63,  # * mV,
        "Vth": -35.53,  # *mV, # np.random.normal(loc=-35.53, scale=4.0, size=NN) * mV,  # -35.53*mV,
        "Vpeak": 21.72,  # * mV,
        "Vmin": -48.7,  # * mV,
        "a": 0.005,  # * ms ** -1,
        "b": 0.22,  # * mS,
        "d": 2,  # * pA,

        "Iext": 800,  # pA
    }

    # Словарь с константами
    cauchy_dencity_params = {
        'Delta_eta': 80,  # 0.02,
        'bar_eta': 0.0,  # 0.191,
    }

    dim_izh_params = dim_izh_params | cauchy_dencity_params
    izh_params = izhs_lib.dimensional_to_dimensionless(dim_izh_params)
    izh_params['dts_non_dim'] = izhs_lib.transform_T(dt_dim, dim_izh_params['Cm'], dim_izh_params['k'], dim_izh_params['Vrest'])

    for key, val in izh_params.items():
        izh_params[key] = np.zeros(NN, dtype=np.float32) + val

    ## synaptic static variables
    tau_d = 6.02  # ms
    tau_r = 359.8  # ms
    tau_f = 21.0  # ms
    Uinc = 0.25

    gsyn_max = np.zeros(shape=(NN+Ninps, NN), dtype=np.float32)
    gsyn_max[0, 1] = 20
    gsyn_max[1, 0] = 15



    pconn = np.zeros(shape=(NN+Ninps, NN), dtype=np.float32)
    pconn[0, 1] = 1
    pconn[1, 0] = 1

    Erev = np.zeros(shape=(NN+Ninps, NN), dtype=np.float32) - 75
    e_r = izhs_lib.transform_e_r(Erev, dim_izh_params['Vrest'])

    izh_params['gsyn_max'] = gsyn_max
    izh_params['pconn'] = pconn
    izh_params['e_r'] = np.zeros_like(gsyn_max) + e_r
    izh_params['tau_d'] = np.zeros_like(gsyn_max) + tau_d
    izh_params['tau_r'] = np.zeros_like(gsyn_max) + tau_r
    izh_params['tau_f'] = np.zeros_like(gsyn_max) + tau_f
    izh_params['Uinc'] = np.zeros_like(gsyn_max) + Uinc


    t = np.arange(0, duration, dt_dim, dtype=np.float32)
    t = t.reshape(1, -1, 1)

    firings_inputs = np.zeros(shape=(1, t.size, Ninps), dtype=np.float32)

    # for key, val in izh_params.items():
    #     print(key, "\n", val)



    model = MeanFieldNetwork(izh_params, dt_dim=dt_dim, use_input=True)

    rates, hist_states = model.predict(firings_inputs)

    A = hist_states[-1]

    gsyn_12 = A[:, 0, 1] * gsyn_max[0, 1]
    gsyn_21 = A[:, 1, 0] * gsyn_max[1, 0]

    print(A.shape)
    print(rates.shape)
    print(hist_states[1].shape)

    rates = rates.reshape(-1, NN)
    t = t.ravel()

    fig, axes = plt.subplots(nrows=4)
    axes[0].plot(t, rates)
    axes[1].plot(t, hist_states[1].reshape(-1, NN))
    axes[2].plot(t, hist_states[2].reshape(-1, NN))
    axes[3].plot(t, gsyn_12)
    axes[3].plot(t, gsyn_21)



    plt.show()