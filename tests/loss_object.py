import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

bessel_i0 = tf.math.bessel_i0
sqrt = tf.math.sqrt
exp = tf.math.exp
cos = tf.math.cos
sin = tf.math.sin
maximum = tf.math.maximum
minimum = tf.math.minimum
abs = tf.math.abs
logical_and = tf.math.logical_and
logical_not = tf.math.logical_not
argmax = tf.math.argmax

PI = np.pi


class CommonGenerator(tf.Module):
    def __init__(self, params, mask=None):
        super(CommonGenerator, self).__init__()
        self.mask = mask

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

class SpatialThetaGenerators(CommonGenerator):
    def __init__(self, params, mask=None):
        super(SpatialThetaGenerators, self).__init__(params, mask)
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


        self.ThetaFreq = tf.constant(ThetaFreq, dtype=tf.float64)
        self.OutPlaceFiringRate = tf.constant(OutPlaceFiringRate, dtype=tf.float64)
        self.OutPlaceThetaPhase = tf.constant(OutPlaceThetaPhase, dtype=tf.float64)
        self.R = tf.constant(Rs, dtype=tf.float64)
        self.InPlacePeakRate = tf.constant(InPlacePeakRate, dtype=tf.float64)
        self.CenterPlaceField = tf.constant(CenterPlaceField, dtype=tf.float64)
        self.SigmaPlaceField = tf.constant(SigmaPlaceField, dtype=tf.float64)
        self.SlopePhasePrecession = tf.constant(SlopePhasePrecession, dtype=tf.float64)
        self.PrecessionOnset = tf.constant(PrecessionOnset, dtype=tf.float64)

    def precomute(self):
        self.kappa = self.r2kappa(self.R)

        self.mult4time = tf.constant(2 * PI * self.ThetaFreq * 0.001, dtype=tf.float64)

        I0 = bessel_i0(self.kappa)
        self.normalizator = self.OutPlaceFiringRate / I0
    def get_firings(self, t):
        ampl4gauss = 2 * (self.InPlacePeakRate - self.OutPlaceFiringRate) / (self.OutPlaceFiringRate + 1) #  range [-1, inf]

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

        return firings

    def get_loss(self, simulated_firings, t):
        t = tf.reshape(t, shape=(-1, 1))
        target_firings = self.get_firings(t)
        target_firings = tf.transpose(target_firings)
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=0)
        loss = tf.keras.losses.logcosh(target_firings, selected_firings)
        return loss
#########################################################################
class VonMisesLoss(CommonGenerator):
    def __init__(self, params, mask=None, sigma_low=0.2, sigma_hight=0.01, dt=0.1):
        super(VonMisesLoss, self).__init__(params, mask)


        Rs = []
        ThetaFreq = []
        ThetaPhase = []
        MeanFiringRate = []


        for p in params:

            Rs.append(p["R"])
            ThetaFreq.append( p["ThetaFreq"] )
            ThetaPhase.append( p["ThetaPhase"] )
            MeanFiringRate.append(p["MeanFiringRate"] )

        self.ThetaFreq = tf.constant(ThetaFreq, dtype=tf.float64)
        self.MeanFiringRate = tf.constant(MeanFiringRate, dtype=tf.float64)
        self.ThetaPhase = tf.constant(ThetaPhase, dtype=tf.float64)
        self.R = tf.constant(Rs, dtype=tf.float64)

        tw = tf.range(-3*sigma_low, 3-sigma_low, 0.001*dt, dtype=tf.float64)

        self.gauss_low =  exp(-0.5 * (tw/sigma_low)**2 )
        self.gauss_low = self.gauss_low / tf.reduce_sum(self.gauss_low)
        self.gauss_low = tf.reshape(self.gauss_low, shape=(-1, 1, 1))

        self.gauss_high =  exp(-0.5 * (tw/sigma_hight)**2 )
        self.gauss_high = self.gauss_high / tf.reduce_sum(self.gauss_high)
        self.gauss_high = tf.reshape(self.gauss_high, shape=(-1, 1, 1))

    def precomute(self):
        self.kappa = self.r2kappa(self.R)

        self.mult4time = tf.constant(2 * PI * self.ThetaFreq * 0.001, dtype=tf.float64)

        I0 = bessel_i0(self.kappa)
        self.normalizator = self.MeanFiringRate / I0 # * 0.001

    def get_firings(self, t):
        firings = self.normalizator * exp(self.kappa * cos(self.mult4time * t - self.ThetaPhase) )
        return firings

    def get_loss(self, simulated_firings, t):
        t = tf.reshape(t, shape=(-1, 1))
        target_firings = self.get_firings(t)
        target_firings = tf.transpose(target_firings)
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=0)


        low_component = tf.nn.conv1d(selected_firings, self.gauss_low, stride=1, padding='SAME', data_format="NWC")
        filtered_firings = (selected_firings - low_component) + tf.reduce_mean(low_component)
        filtered_firings = tf.nn.conv1d(filtered_firings, self.gauss_high, stride=1, padding='SAME', data_format="NWC")
        loss = tf.keras.losses.cosine_similarity(filtered_firings, target_firings)

        robast_mean = exp(tf.reduce_mean(log(simulated_firings), axis=-1)) #!!!!

        loss = loss + tf.keras.losses.MSE(self.MeanFiringRate, robast_mean)

        return loss
#########################################################################
class LockingLoss(CommonGenerator):
    def __init__(self, params, mask=None, dt=0.1):
        super(LockingLoss, self).__init__(params, mask)

        Rs = []
        ThetaFreq = []

        LowFiringRateBound = []
        HighFiringRateBound = []

        for p in params:
            Rs.append(p["R"])
            ThetaFreq.append(p["ThetaFreq"])
            LowFiringRateBound.append(p["LowFiringRateBound"])
            HighFiringRateBound.append(p["HighFiringRateBound"])

        self.ThetaFreq = tf.constant(ThetaFreq, dtype=tf.float64)
        self.LowFiringRateBound = tf.constant(LowFiringRateBound, dtype=tf.float64)
        self.HighFiringRateBound = tf.constant(HighFiringRateBound, dtype=tf.float64)
        self.R = tf.constant(Rs, dtype=tf.float64)

    def get_loss(self, simulated_firings, t):
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=0)
        t = tf.reshape(t, shape=(-1, 1))

        theta_phases = 2 * PI * 0.001 * t * self.ThetaFreq
        real = cos(theta_phases)
        imag = sin(theta_phases)

        selected_firings = selected_firings / tf.math.reduce_sum(selected_firings, axis=-1)  #

        Rsim = sqrt(tf.math.reduce_sum(selected_firings * real) ** 2 + tf.math.reduce_sum(selected_firings * imag) ** 2)

        loss = tf.keras.losses.MSE(Rsim, self.R)

        loss += tf.keras.activations.relu( selected_firings, threshold=self.HighFiringRateBound)
        loss += tf.keras.activations.relu( self.LowFiringRateBound-selected_firings)

        return loss



#############################################################################
default_params = {
            "R": 0.25,
            "OutPlaceFiringRate" : 5.0,
            "OutPlaceThetaPhase" : 2.0, #3.14,
            "InPlacePeakRate" : 15.0,
            "CenterPlaceField" : 5000.0,
            "SigmaPlaceField" : 500,
            "SlopePhasePrecession" : 0.0, #np.deg2rad(10)*10 * 0.001,
            "ThetaFreq" : 8.0,
            "PrecessionOnset" : -1.57,
}

von_mises_params = {
            "R": 0.25,
            "ThetaPhase" : 3.14,
            "ThetaFreq" : 8.0,
            "MeanFiringRate" : 5.0,
}

params = []

#mask = tf.constant( [True, True, False, False, True], dtype=tf.dtypes.bool)
mask = tf.constant( [True, ], dtype=tf.dtypes.bool)

for _ in range(tf.reduce_sum(tf.cast(mask, dtype=tf.dtypes.int16))):
    params.append(default_params)


t = tf.range(0, 10000.0, 0.1, dtype=tf.float64)

genrators = SpatialThetaGenerators(params, mask)
genrators.precomute()

vonmises_gen = VonMisesLoss([von_mises_params, ], mask)
vonmises_gen.precomute()
simulated_firings = genrators.get_firings(tf.reshape(t, shape=(-1, 1))) #exp(-0.5 * ((t - 5000)/300)**2) * cos(2*PI*6*t*0.001) #
# tf.random.uniform( shape=(5, tf.size(t)), maxval=1.0, dtype=tf.dtypes.float64))
#loss = genrators.get_loss(simulated_firings, t)

scale = 0.2
w0 = 6
tw = tf.range(-1.5, 1.5, 0.0001, dtype=tf.float64)
morlet = exp(-0.5 * (tw/scale)**2 )  #* PI**(-0.25) *  cos(2*PI*w0*tw/scale)
morlet = morlet / tf.reduce_sum(morlet)

simulated_firings = tf.reshape(simulated_firings, shape=(1, -1, 1))
morlet = tf.reshape(morlet, shape=(-1, 1, 1))

W = tf.nn.conv1d(simulated_firings, morlet, stride=1, padding='SAME', data_format="NWC")

W = tf.reshape(W, shape=(-1, ))
simulated_firings = tf.reshape(simulated_firings, shape=(-1, ))
#print(loss)

# simulated_firings = np.asarray(simulated_firings).ravel()
morlet = np.asarray(morlet).ravel()
#
# Wnp = np.convolve(simulated_firings, morlet, mode="same")

filtered_firings = (simulated_firings - W) + tf.reduce_mean(W)
#filtered_firings = tf.nn.softmax(0.01 * (simulated_firings - W))

target_firings = vonmises_gen.get_firings(t)
#target_firings = tf.nn.softmax(0.01 * target_firings )


#print(tf.reduce_sum(filtered_firings*target_firings))



l = tf.keras.losses.cosine_similarity(filtered_firings, target_firings)
print(l)

print(tf.reduce_mean(simulated_firings))
robast_mean = exp( tf.reduce_mean( log(simulated_firings) ) )
print(robast_mean)

fig, axes = plt.subplots(nrows=3)
axes[0].plot(t, simulated_firings)
axes[1].plot(tw, morlet)
axes[0].plot(t, W, linewidth=1)
plt.plot(t, filtered_firings, linewidth=2)
plt.plot(t, target_firings, linewidth=2)
plt.show()
