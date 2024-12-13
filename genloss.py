import numpy as np
import tensorflow as tf
import myconfig

tf.keras.backend.set_floatx(myconfig.DTYPE)

bessel_i0 = tf.math.bessel_i0
sqrt = tf.math.sqrt
exp = tf.math.exp
log = tf.math.log
cos = tf.math.cos
sin = tf.math.sin
maximum = tf.math.maximum
minimum = tf.math.minimum
abs = tf.math.abs
logical_and = tf.math.logical_and
logical_not = tf.math.logical_not
argmax = tf.math.argmax

PI = np.pi


class SimplestKeepLayer(tf.keras.layers.Layer):
    def __init__(self, params):
        super(SimplestKeepLayer, self).__init__()
        self.targets_vals = tf.constant(params, dtype=myconfig.DTYPE)

    def call(self, t):
        return self.targets_vals

    def get_config(self):
        config = super().get_config()
        config.update({
            'targets_vals': self.targets_vals,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config['targets_vals'])

class RILayer(tf.keras.layers.Layer):
    def __init__(self, params):
        super(RILayer, self).__init__()

        R = []
        ThetaPhase = []

        for p in params:
            R.append(p["R"])
            ThetaPhase.append(p["ThetaPhase"])

        self.R = tf.constant(R, dtype=myconfig.DTYPE)
        self.ThetaPhase = tf.constant(ThetaPhase, dtype=myconfig.DTYPE)

        imag = R * sin(ThetaPhase)
        real = R * cos(ThetaPhase)

        self.targets_vals = tf.stack([real, imag], axis=1)

    def call(self, t):
        return self.targets_vals

    def get_config(self):
        config = super().get_config()
        config.update({
            'R': self.R,
            'ThetaPhase': self.ThetaPhase,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)





    #### inputs generators
class CommonGenerator(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(CommonGenerator, self).__init__(**kwargs)


    # def build(self):
    #     pass

    def r2kappa(self, R):
        """
        recalulate kappa from R for von Misses function
        """

        # if R < 0.53:
        #     kappa = 2 * R + R ** 3 + 5 / 6 * R ** 5
        #
        # elif R >= 0.53 and R < 0.85:
        #     kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
        #
        # elif R >= 0.85:
        #     kappa = 1 / (3 * R - 4 * R ** 2 + R ** 3)
        kappa = tf.where(R < 0.53,  2 * R + R**3 + 5 / 6 * R**5, 0.0)
        kappa = tf.where(logical_and(R >= 0.53, R < 0.85),  -0.4 + 1.39 * R + 0.43 / (1 - R), kappa)
        kappa = tf.where(R >= 0.85,  1 / (3 * R - 4 * R**2 + R**3), kappa)
        return kappa

    def get_config(self):
        config = super().get_config()
        return config

class VonMissesGenerator(CommonGenerator):
    def __init__(self, params):
        super(VonMissesGenerator, self).__init__(params)

        ThetaFreq = []
        FiringRate = []
        Rs = []
        ThetaPhase = []


        self.n_outs = len(params)

        for p in params:
            FiringRate.append(p["MeanFiringRate"])
            Rs.append(p["R"]),
            ThetaFreq.append(p["ThetaFreq"])
            ThetaPhase.append(p["ThetaPhase"])

        self.ThetaFreq = tf.reshape(tf.constant(ThetaFreq, dtype=myconfig.DTYPE), [1, -1])
        self.FiringRate = tf.reshape(tf.constant(FiringRate, dtype=myconfig.DTYPE), [1, -1])
        self.R = tf.reshape(tf.constant(Rs, dtype=myconfig.DTYPE), [1, -1])
        self.ThetaPhase = tf.reshape(tf.constant(ThetaPhase, dtype=myconfig.DTYPE), [1, -1])

    def build(self):
        input_shape = (None, 1)

        super(VonMissesGenerator, self).build(input_shape)
        self.kappa = self.r2kappa(self.R)
        tmp = 2 * PI * self.ThetaFreq * 0.001
        self.mult4time = tf.constant(tmp, dtype=myconfig.DTYPE)
        I0 = bessel_i0(self.kappa)
        self.normalizator = self.FiringRate / I0


        self.built = True

    def call(self, t):
        firings = self.normalizator * exp(self.kappa * cos( self.mult4time * t - self.ThetaPhase))

        #firings = firings  # / (0.001 * dt)

        firings = tf.reshape(firings, shape=(1, tf.shape(firings)[0], tf.shape(firings)[1]  ))

        return firings

    # Реализация метода get_config
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_outs': self.n_outs,
            'ThetaFreq': self.ThetaFreq.numpy().tolist(),
            'FiringRate': self.FiringRate.numpy().tolist(),
            'R': self.R.numpy().tolist(),
            'ThetaPhase': self.ThetaPhase.numpy().tolist(),
        })
        return config

    # Реализация метода from_config
    @classmethod
    def from_config(cls, config):
        return cls(config)

class SpatialThetaGenerators(CommonGenerator):
    def __init__(self, params, **kwargs):
        super(SpatialThetaGenerators, self).__init__(params, **kwargs)
        self.ALPHA = 5.0

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


        self.ThetaFreq = tf.reshape( tf.constant(ThetaFreq, dtype=myconfig.DTYPE), [1, -1])
        self.OutPlaceFiringRate = tf.reshape( tf.constant(OutPlaceFiringRate, dtype=myconfig.DTYPE), [1, -1])
        self.OutPlaceThetaPhase = tf.reshape( tf.constant(OutPlaceThetaPhase, dtype=myconfig.DTYPE), [1, -1])
        self.R = tf.reshape( tf.constant(Rs, dtype=myconfig.DTYPE), [1, -1])
        self.InPlacePeakRate = tf.reshape( tf.constant(InPlacePeakRate, dtype=myconfig.DTYPE), [1, -1])
        self.CenterPlaceField = tf.reshape(  tf.constant(CenterPlaceField, dtype=myconfig.DTYPE), [1, -1])
        self.SigmaPlaceField = tf.reshape( tf.constant(SigmaPlaceField, dtype=myconfig.DTYPE), [1, -1])
        self.SlopePhasePrecession = tf.reshape( tf.constant(SlopePhasePrecession, dtype=myconfig.DTYPE), [1, -1])
        self.PrecessionOnset = tf.reshape( tf.constant(PrecessionOnset, dtype=myconfig.DTYPE), [1, -1])

        Mask = tf.math.is_nan(self.PrecessionOnset)
        self.PrecessionOnset = tf.where(Mask, 0., self.PrecessionOnset )
        self.PhaseOnsetMask = tf.cast(Mask, dtype=myconfig.DTYPE) - 1.0

    def build(self):
        input_shape = (None, 1)

        super(SpatialThetaGenerators, self).build(input_shape)

        self.kappa = self.r2kappa(self.R)

        self.mult4time = 2 * PI * self.ThetaFreq * 0.001 #tf.constant(tmp, dtype=myconfig.DTYPE)

        I0 = bessel_i0(self.kappa)
        self.normalizator = self.OutPlaceFiringRate / I0


        self.built = True


    def call(self, t):
        ampl4gauss = 2 * (self.InPlacePeakRate - self.OutPlaceFiringRate) / (self.OutPlaceFiringRate + 1)
        multip = (1 + ampl4gauss * exp(-0.5 * ((t - self.CenterPlaceField) / self.SigmaPlaceField) ** 2))

        start_place = t - self.CenterPlaceField - 3 * self.SigmaPlaceField
        end_place = t - self.CenterPlaceField + 3 * self.SigmaPlaceField



        inplace = 0.25 * (1.0 - (start_place / (self.ALPHA + tf.math.abs(start_place)))) * (
                1.0 + end_place / (self.ALPHA + abs(end_place)))

        precession = self.SlopePhasePrecession * inplace
        phases = self.OutPlaceThetaPhase * (1 - inplace * self.PhaseOnsetMask ) + self.PrecessionOnset * inplace * self.PhaseOnsetMask


        firings = self.normalizator * exp(self.kappa * cos((self.mult4time + precession) * t - phases))

        firings = multip * firings  # / (0.001 * dt)
        firings = tf.reshape(firings, shape=(1, tf.shape(firings)[1], tf.shape(firings)[2]))

        return firings

    def get_config(self):
        config = super().get_config()
        ThetaFreq = self.ThetaFreq.numpy().tolist()
        R = self.R.numpy().tolist()
        OutPlaceFiringRate = self.OutPlaceFiringRate.numpy().tolist()
        OutPlaceThetaPhase = self.OutPlaceThetaPhase.numpy().tolist()
        InPlacePeakRate = self.InPlacePeakRate.numpy().tolist()
        CenterPlaceField = self.CenterPlaceField.numpy().tolist()
        SigmaPlaceField  = self.SigmaPlaceField.numpy().tolist()
        SlopePhasePrecession = self.SlopePhasePrecession.numpy().tolist()
        PrecessionOnset = self.PrecessionOnset.numpy().tolist()

        myparams = []
        for idx in range(self.n_outs):
            p = {
                "ThetaFreq" : ThetaFreq[0][idx],
                "R" : R[0][idx],
                "OutPlaceFiringRate" : OutPlaceFiringRate[0][idx],
                "OutPlaceThetaPhase" : OutPlaceThetaPhase[0][idx],
                "InPlacePeakRate" : InPlacePeakRate[0][idx],
                "CenterPlaceField" : CenterPlaceField[0][idx],
                "SigmaPlaceField" : SigmaPlaceField[0][idx],
                "SlopePhasePrecession" : SlopePhasePrecession[0][idx],
                "PrecessionOnset" : PrecessionOnset[0][idx],
            }
            myparams.append(p)

        config.update({
            'myparams' : myparams,
        })
        return config

    # Реализация метода from_config
    @classmethod
    def from_config(cls, config):
        params = config.pop('myparams')
        return cls(params, **config)


#########################################################################
##### Output processing classes #########################################

class CommonOutProcessing(tf.keras.layers.Layer):
    def __init__(self, mask, **kwargs):
        super(CommonOutProcessing, self).__init__(**kwargs)
        #
        # self.inputs_size = len(params)
        #
        # if mask is None:
        #     mask = np.ones(self.inputs_size, dtype='bool')

        self.mask = tf.constant(mask, dtype=tf.dtypes.bool)
        self.input_size = tf.size(self.mask)
        self.n_selected = tf.reduce_sum( tf.cast(self.mask, dtype=tf.int64)  )

    def build(self):
        input_shape = (1, None, self.input_size)
        super(CommonOutProcessing, self).build(input_shape)
        self.built = True


    def call(self, firings):
        selected_firings = tf.boolean_mask(firings, self.mask, axis=2)
        return selected_firings

    def get_config(self):
        config = super().get_config()
        config.update({
            'mask': self.mask.numpy().tolist(),
        })
        return config

    # Реализация метода from_config
    @classmethod
    def from_config(cls, config):
        mask = config.pop('mask')
        return cls(mask, **config)


# class FrequencyFilter(CommonOutProcessing):
#     def __init__(self, mask, minfreq=3, maxfreq=8, dfreq=1, dt=0.1, **kwargs):
#     # def __init__(self, mask, sigma_low=0.2, sigma_hight=0.01, dt=0.1):
#         super(FrequencyFilter, self).__init__(mask, **kwargs)
#
#         self.omega0 = 6.0 # w0 of mortet
#         freqs = tf.range(minfreq, maxfreq, dfreq, dtype=myconfig.DTYPE)
#         self.scales = self.omega0 / freqs
#         self.dt = tf.constant(0.001 * dt, dtype=myconfig.DTYPE) # dt = 0.001 * dt : convert ms to sec
#
#
#         # tw = tf.range(-3*sigma_low, 3*sigma_low, 0.001*dt, dtype=myconfig.DTYPE)
#         #
#         # self.gauss_low = exp(-0.5 * (tw/sigma_low)**2 )
#         # self.gauss_low = self.gauss_low / tf.reduce_sum(self.gauss_low)
#         # self.gauss_low = tf.reshape(self.gauss_low, shape=(-1, 1, 1))
#         # self.gauss_low = tf.concat( int(self.n_selected) *(self.gauss_low, ), axis=2)
#         #
#         # self.gauss_high =  exp(-0.5 * (tw/sigma_hight)**2 )
#         # self.gauss_high = self.gauss_high / tf.reduce_sum(self.gauss_high)
#         # self.gauss_high = tf.reshape(self.gauss_high, shape=(-1, 1, 1))
#         # self.gauss_high = tf.concat( int(self.n_selected) * (self.gauss_high,), axis=2)
#
#     def build(self):
#         super(FrequencyFilter, self).build()
#
#     def fftfreqs(self, n, dt):
#         val = 1.0 / (tf.cast(n, dtype=myconfig.DTYPE) * dt)
#
#         N = tf.where((n % 2) == 0, n / 2 + 1, (n - 1) / 2)
#         p1 = tf.range(0, N, dtype=myconfig.DTYPE)
#         N = tf.where((n % 2) == 0, -n / 2, -(n - 1) / 2)
#         p2 = tf.range(N, -1, dtype=myconfig.DTYPE)
#         results = tf.concat([p1, p2], axis=0)
#
#         return results * val
#
#
#     def call(self, simulated_firings):
#
#         selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=2)
#
#         # !!!!!!!!!!!!!!!!
#         # low_component = tf.nn.conv1d(selected_firings, self.gauss_low, stride=1, padding='SAME') # , data_format="NWC"
#         # filtered_firings = (selected_firings - low_component) + tf.reduce_mean(low_component)
#         # filtered_firings = tf.nn.conv1d(filtered_firings, self.gauss_high, stride=1, padding='SAME') # , data_format="NWC"
#
#         coeff_map = ()
#
#         signal_FT = tf.signal.fft(selected_firings)
#         omegas = self.fftfreqs(tf.shape(selected_firings)[1], self.dt)
#
#         for idx, s in enumerate(self.scales):
#             morlet_FT = PI ** (-0.25) * tf.math.exp(-0.5 * (s * omegas - self.omega0)**2)
#             morlet_FT = tf.cast(morlet_FT, dtype=tf.complex64)
#
#             coeff = tf.signal.ifft(signal_FT * morlet_FT) / tf.cast(tf.math.sqrt(s), dtype=tf.complex64)
#             coeff_map = coeff_map + (coeff,)
#
#         coeff_map = tf.stack(coeff_map, axis=1)
#
#
#         filtered_firings = tf.reduce_sum(tf.math.real(coeff_map), axis=1)
#         filtered_firings = filtered_firings - tf.reduce_min(filtered_firings)
#
#         return filtered_firings

#########################################################################
class PhaseLockingOutput(CommonOutProcessing):
    def __init__(self, mask=None, ThetaFreq=5.0, dt=0.1, **kwargs):
        super(PhaseLockingOutput, self).__init__(mask, **kwargs)


        self.ThetaFreq = tf.constant(ThetaFreq, dtype=myconfig.DTYPE)
        self.dt = tf.constant(dt, dtype=myconfig.DTYPE)



    def compute_fourie_trasform(self, selected_firings):
        t_max = tf.cast(tf.shape(selected_firings)[1], dtype=myconfig.DTYPE) * self.dt

        t = tf.range(0, t_max, self.dt)
        t = tf.reshape(t, shape=(-1, 1))

        theta_phases = 2 * PI * 0.001 * t * self.ThetaFreq
        real = cos(theta_phases)
        imag = sin(theta_phases)

        normed_firings = selected_firings / (tf.math.reduce_sum(selected_firings, axis=1) + 0.00000000001)

        real_sim = tf.reduce_sum(normed_firings * real, axis=1)
        imag_sim = tf.reduce_sum(normed_firings * imag, axis=1)

        return real_sim, imag_sim

    def call(self, simulated_firings):
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=2)
        real_sim, imag_sim = self.compute_fourie_trasform(selected_firings)
        Rsim = sqrt(real_sim**2 + imag_sim**2 + 0.0000001)
        Rsim = tf.reshape(Rsim, shape=(1, 1, -1))
        return Rsim

    def get_config(self):
        config = super().get_config()
        config.update({
            'ThetaFreq': self.ThetaFreq.numpy().tolist(),
            'dt': self.dt.numpy().tolist(),
        })
        return config

    # Реализация метода from_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)



class PhaseLockingOutputWithPhase(PhaseLockingOutput):

    def __init__(self, mask=None, ThetaFreq=5.0, dt=0.1, **kwargs):
        super(PhaseLockingOutputWithPhase, self).__init__(mask=mask, ThetaFreq=ThetaFreq, dt=dt, **kwargs)

        # phases = []
        # for p in params:
        #     phases.append(p["ThetaPhase"])

        #self.phases = tf.constant(phases, dtype=myconfig.DTYPE)

    def call(self, simulated_firings):
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=2)
        real_sim, imag_sim = self.compute_fourie_trasform(selected_firings)

        output = tf.stack([real_sim, imag_sim], axis=1)

        output = tf.reshape(output, shape=(1, 2, -1))
        return output


    def get_config(self):
        config = super().get_config()
        config.update({
            'ThetaFreq': self.ThetaFreq.numpy().tolist(),
            'dt': self.dt.numpy().tolist(),
        })
        return config

    # Реализация метода from_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)



class RobastMeanOut(CommonOutProcessing):

    def __init__(self, mask=None, **kwargs):
        super(RobastMeanOut, self).__init__(mask, **kwargs)

    def call(self, simulated_firings):
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=2)

        robast_mean = exp(tf.reduce_mean(log(selected_firings + 0.0001), axis=1))
        robast_mean = tf.reshape(robast_mean, shape=(1, 1, -1))

        return robast_mean

    def get_config(self):
        config = super().get_config()
        config.update({
            'mask': self.mask.numpy().tolist(),
        })
        return config

########################################################################################################################
##### outputs regulizers
class FiringsMeanOutRanger(tf.keras.regularizers.Regularizer):
    def __init__(self, LowFiringRateBound=0.1, HighFiringRateBound=90.0, strength=10):
        self.LowFiringRateBound = tf.convert_to_tensor(LowFiringRateBound)
        self.LowFiringRateBound = tf.reshape(self.LowFiringRateBound, shape=(1, 1, 1, -1))
        self.HighFiringRateBound = tf.convert_to_tensor(HighFiringRateBound)
        self.HighFiringRateBound = tf.reshape(self.HighFiringRateBound, shape=(1, 1, 1, -1))


        self.rw = strength

    def __call__(self, x):
        loss_add = tf.reduce_sum( tf.nn.relu( x - self.HighFiringRateBound) )
        loss_add += tf.reduce_sum( tf.nn.relu( self.LowFiringRateBound - x) )
        return self.rw * loss_add


    def get_config(self):
        config = {
            "LowFiringRateBound": self.LowFiringRateBound.numpy().ravel().tolist(),
            "HighFiringRateBound": self.HighFiringRateBound.numpy().ravel().tolist(),
            "strength": self.rw,
        }

        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Decorrelator(tf.keras.regularizers.Regularizer):
    def __init__(self, strength=0.1):
        self.strength = strength

    def __call__(self, x):
        Ntimesteps = tf.cast(tf.shape(x)[1], dtype=myconfig.DTYPE)
        x = tf.reshape(x, shape=(tf.shape(x)[1], tf.shape(x)[2]))

        Xcentered = x - tf.reduce_mean(x, axis=0, keepdims=True)
        Xcentered = Xcentered / (tf.math.sqrt(tf.reduce_mean(Xcentered ** 2, axis=0, keepdims=True)) + 0.0000000001)
        corr_matrix = (tf.transpose(Xcentered) @ Xcentered) / Ntimesteps

        return self.strength * tf.reduce_mean(corr_matrix**2)

    # Метод для получения конфигурации
    def get_config(self):
        return {"strength": self.strength}

    # Статический метод для создания экземпляра класса из конфигурации
    @classmethod
    def from_config(cls, config):
        return cls(**config)