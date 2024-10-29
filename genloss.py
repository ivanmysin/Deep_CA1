import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float32')

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
        self.targets_vals = tf.constant(params, dtype=tf.float32)

    def call(self, t):
        return self.targets_vals



#### inputs generators
class CommonGenerator(tf.keras.layers.Layer):
    def __init__(self, params):
        super(CommonGenerator, self).__init__()


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

        self.ThetaFreq = tf.reshape(tf.constant(ThetaFreq, dtype=tf.float32), [1, -1])
        self.FiringRate = tf.reshape(tf.constant(FiringRate, dtype=tf.float32), [1, -1])
        self.R = tf.reshape(tf.constant(Rs, dtype=tf.float32), [1, -1])
        self.ThetaPhase = tf.reshape(tf.constant(ThetaPhase, dtype=tf.float32), [1, -1])

    def build(self):
        input_shape = (None, 1)

        super(VonMissesGenerator, self).build(input_shape)
        self.kappa = self.r2kappa(self.R)
        tmp = 2 * PI * self.ThetaFreq * 0.001
        self.mult4time = tf.constant(tmp, dtype=tf.float32)
        I0 = bessel_i0(self.kappa)
        self.normalizator = self.FiringRate / I0


        self.built = True

    def call(self, t):
        firings = self.normalizator * exp(self.kappa * cos( self.mult4time * t - self.ThetaPhase))

        #firings = firings  # / (0.001 * dt)

        firings = tf.reshape(firings, shape=(1, tf.shape(firings)[0], tf.shape(firings)[1]  ))

        return firings


class SpatialThetaGenerators(CommonGenerator):
    def __init__(self, params):
        super(SpatialThetaGenerators, self).__init__(params)
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


        self.ThetaFreq = tf.reshape( tf.constant(ThetaFreq, dtype=tf.float32), [1, -1])
        self.OutPlaceFiringRate = tf.reshape( tf.constant(OutPlaceFiringRate, dtype=tf.float32), [1, -1])
        self.OutPlaceThetaPhase = tf.reshape( tf.constant(OutPlaceThetaPhase, dtype=tf.float32), [1, -1])
        self.R = tf.reshape( tf.constant(Rs, dtype=tf.float32), [1, -1])
        self.InPlacePeakRate = tf.reshape( tf.constant(InPlacePeakRate, dtype=tf.float32), [1, -1])
        self.CenterPlaceField = tf.reshape(  tf.constant(CenterPlaceField, dtype=tf.float32), [1, -1])
        self.SigmaPlaceField = tf.reshape( tf.constant(SigmaPlaceField, dtype=tf.float32), [1, -1])
        self.SlopePhasePrecession = tf.reshape( tf.constant(SlopePhasePrecession, dtype=tf.float32), [1, -1])
        self.PrecessionOnset = tf.reshape( tf.constant(PrecessionOnset, dtype=tf.float32), [1, -1])

    def build(self):
        input_shape = (None, 1)

        super(SpatialThetaGenerators, self).build(input_shape)

        self.kappa = self.r2kappa(self.R)

        tmp = 2 * PI * self.ThetaFreq * 0.001
        self.mult4time = tf.constant(tmp, dtype=tf.float32)

        I0 = bessel_i0(self.kappa)
        self.normalizator = self.OutPlaceFiringRate / I0


        self.built = True


    def call(self, t):
        ampl4gauss = 2 * (self.InPlacePeakRate - self.OutPlaceFiringRate) / (self.OutPlaceFiringRate + 1)
        multip = (1 + ampl4gauss * exp(-0.5 * ((t - self.CenterPlaceField) / self.SigmaPlaceField) ** 2))
        start_place = t - self.CenterPlaceField - 3 * self.SigmaPlaceField
        end_place = t - self.CenterPlaceField + 3 * self.SigmaPlaceField



        inplace = 0.25 * (1.0 - (start_place / (self.ALPHA + np.abs(start_place)))) * (
                1.0 + end_place / (self.ALPHA + abs(end_place)))

        precession = self.SlopePhasePrecession * inplace
        #phases = self.OutPlaceThetaPhase * (1 - inplace) + self.PrecessionOnset * inplace #!!!!
        phases = self.OutPlaceThetaPhase

        firings = self.normalizator * exp(self.kappa * cos((self.mult4time + precession) * t - phases))

        firings = multip * firings  # / (0.001 * dt)
        firings = tf.reshape(firings, shape=(1, tf.shape(firings)[0], tf.shape(firings)[1]))

        return firings

    # def get_loss(self, simulated_firings, t):
    #     t = tf.reshape(t, shape=(-1, 1))
    #     target_firings = self.get_firings(t)
    #     target_firings = tf.reshape(target_firings, shape=(1, tf.size(t), tf.size(self.ThetaFreq)))
    #     selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=2)
    #     loss = tf.reduce_mean( tf.keras.losses.logcosh(target_firings, selected_firings) )
    #     return loss


#########################################################################
##### Output processing classes #########################################

class CommonOutProcessing(tf.keras.layers.Layer):
    def __init__(self, mask):
        super(CommonOutProcessing, self).__init__()
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




class FrequencyFilter(CommonOutProcessing):
    def __init__(self, mask, sigma_low=0.2, sigma_hight=0.01, dt=0.1):
        super(FrequencyFilter, self).__init__(mask)


        # Rs = []
        # ThetaFreq = []
        # ThetaPhase = []
        # MeanFiringRate = []
        #
        #
        # for p in params:
        #
        #     Rs.append(p["R"])
        #     ThetaFreq.append( p["ThetaFreq"] )
        #     ThetaPhase.append( p["ThetaPhase"] )
        #     MeanFiringRate.append(p["MeanFiringRate"] )
        #
        # self.ThetaFreq = tf.constant(ThetaFreq, dtype=tf.float32)
        # self.MeanFiringRate = tf.constant(MeanFiringRate, dtype=tf.float32)
        # self.MeanFiringRate = tf.reshape(self.MeanFiringRate, shape=(1, -1))
        # self.ThetaPhase = tf.constant(ThetaPhase, dtype=tf.float32)
        # self.R = tf.constant(Rs, dtype=tf.float32)

        tw = tf.range(-3*sigma_low, 3*sigma_low, 0.001*dt, dtype=tf.float32)

        self.gauss_low = exp(-0.5 * (tw/sigma_low)**2 )
        self.gauss_low = self.gauss_low / tf.reduce_sum(self.gauss_low)
        self.gauss_low = tf.reshape(self.gauss_low, shape=(-1, 1, 1))
        self.gauss_low = tf.concat( int(self.n_selected) *(self.gauss_low, ), axis=2)

        self.gauss_high =  exp(-0.5 * (tw/sigma_hight)**2 )
        self.gauss_high = self.gauss_high / tf.reduce_sum(self.gauss_high)
        self.gauss_high = tf.reshape(self.gauss_high, shape=(-1, 1, 1))
        self.gauss_high = tf.concat( int(self.n_selected) * (self.gauss_high,), axis=2)

    def build(self):
        super(FrequencyFilter, self).build()
        #
        # self.kappa = self.r2kappa(self.R)
        #
        # self.mult4time = tf.constant(2 * PI * self.ThetaFreq * 0.001, dtype=tf.float32)
        #
        # I0 = bessel_i0(self.kappa)
        # self.normalizator = self.MeanFiringRate / I0 # * 0.001

    def call(self, simulated_firings):
        #firings = self.normalizator * exp(self.kappa * cos(self.mult4time * t - self.ThetaPhase) )
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=2)

        # !!!!!!!!!!!!!!!!
        # low_component = tf.nn.conv1d(selected_firings, self.gauss_low, stride=1, padding='SAME') # , data_format="NWC"
        # filtered_firings = (selected_firings - low_component) + tf.reduce_mean(low_component)
        # filtered_firings = tf.nn.conv1d(filtered_firings, self.gauss_high, stride=1, padding='SAME') # , data_format="NWC"

        filtered_firings = selected_firings

        return filtered_firings


    # def get_loss(self, simulated_firings, t):
    #     t = tf.reshape(t, shape=(-1, 1))
    #     target_firings = self.get_firings(t)
    #     target_firings = tf.reshape(target_firings, shape=(1, tf.size(t), tf.size(self.ThetaFreq)))
    #     selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=2)
    #
    #     low_component = tf.nn.conv1d(selected_firings, self.gauss_low, stride=1, padding='SAME', data_format="NWC")
    #     filtered_firings = (selected_firings - low_component) + tf.reduce_mean(low_component)
    #     filtered_firings = tf.nn.conv1d(filtered_firings, self.gauss_high, stride=1, padding='SAME', data_format="NWC")
    #     loss = tf.reduce_mean( tf.keras.losses.cosine_similarity(filtered_firings, target_firings) )
    #
    #
    #     robast_mean = exp(tf.reduce_mean(log(selected_firings), axis=1))
    #     loss = loss + tf.keras.losses.MSE(self.MeanFiringRate, robast_mean)
    #
    #     return loss
#########################################################################
class PhaseLockingOutput(CommonOutProcessing):
    def __init__(self, mask=None, ThetaFreq=5.0, dt=0.1):
        super(PhaseLockingOutput, self).__init__(mask)

        # Rs = []


        # for p in params:
        #     # Rs.append(p["R"])
        #     ThetaFreq.append(p["ThetaFreq"])
        #     # LowFiringRateBound.append(p["LowFiringRateBound"])
        #     # HighFiringRateBound.append(p["HighFiringRateBound"])

        self.ThetaFreq = tf.constant(ThetaFreq, dtype=tf.float32)
        self.dt = tf.constant(dt, dtype=tf.float32)
        # self.LowFiringRateBound = tf.constant(LowFiringRateBound, dtype=tf.float32)
        # self.HighFiringRateBound = tf.constant(HighFiringRateBound, dtype=tf.float32)
        # self.R = tf.constant(Rs, dtype=tf.float32)

    def call(self, simulated_firings):
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=2)

        #print("Nans selected_firings", tf.math.reduce_sum(  tf.cast(tf.math.is_nan(selected_firings), dtype=tf.int32)   )  )

        t_max = tf.cast(tf.shape(simulated_firings)[1], dtype=tf.float32) * self.dt

        t = tf.range(0, t_max, self.dt)
        t = tf.reshape(t, shape=(-1, 1))

        theta_phases = 2 * PI * 0.001 * t * self.ThetaFreq
        real = cos(theta_phases)
        imag = sin(theta_phases)

        normed_firings = selected_firings / (tf.math.reduce_sum(selected_firings, axis=1) + 0.00000000001)
        # print("Nans normed_firings", tf.math.reduce_sum(  tf.cast(tf.math.is_nan(normed_firings), dtype=tf.int32)   )  )


        Rsim = sqrt(tf.math.reduce_sum(normed_firings * real, axis=1)**2 + tf.math.reduce_sum(normed_firings * imag, axis=1)**2 + 0.0000001)
        Rsim = tf.reshape(Rsim, shape=(-1))


        #print("Nans Rsim", tf.math.reduce_sum( tf.cast(tf.math.is_nan(Rsim), dtype=tf.int32)  ))

        # loss = tf.keras.losses.MSE(Rsim, self.R)
        #
        # mean_firings = tf.reduce_mean(selected_firings, axis=1)
        # loss += tf.reduce_sum( tf.nn.relu( mean_firings - self.HighFiringRateBound) )
        # loss += tf.reduce_sum( tf.nn.relu( self.LowFiringRateBound - mean_firings) )

        return Rsim


class RobastMeanOut(CommonOutProcessing):

    def __init__(self, mask=None):
        super(RobastMeanOut, self).__init__(mask)

    def call(self, simulated_firings):
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=2)

        robast_mean = exp(tf.reduce_mean(log(selected_firings + 0.0001), axis=1))
        robast_mean = tf.reshape(robast_mean, shape=(-1))

        return robast_mean


########################################################################################################################
##### outputs regulizers
#@tf.keras.utils.register_keras_serializable(package='Custom', name='l2')
class RobastMeanOutRanger(tf.keras.regularizers.Regularizer):
    def __init__(self, LowFiringRateBound=0.1, HighFiringRateBound=90.0, strength=10):
        self.LowFiringRateBound = LowFiringRateBound
        self.HighFiringRateBound = HighFiringRateBound
        self.rw = strength

    def __call__(self, x):
        loss_add = tf.reduce_sum( tf.nn.relu( x - self.HighFiringRateBound) )
        loss_add += tf.reduce_sum( tf.nn.relu( self.LowFiringRateBound - x) )
        return self.rw * loss_add

    # def get_config(self):
    #   return {'l2': float(self.l2)}
class Decorrelator(tf.keras.regularizers.Regularizer):
    def __init__(self, strength=0.1):
        self.strength = strength

    def __call__(self, x):
        Ntimesteps = tf.cast(tf.shape(x)[1], dtype=tf.float32)
        x = tf.reshape(x, shape=(tf.shape(x)[1], tf.shape(x)[2]))

        Xcentered = x - tf.reduce_mean(x, axis=0, keepdims=True)
        Xcentered = Xcentered / (tf.math.sqrt(tf.reduce_mean(Xcentered ** 2, axis=0, keepdims=True)) + 0.0000000001)
        corr_matrix = (tf.transpose(Xcentered) @ Xcentered) / Ntimesteps

        return self.strength * tf.reduce_mean(corr_matrix**2)