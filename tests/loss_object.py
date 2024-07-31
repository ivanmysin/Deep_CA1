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
        self.normalizator = self.OutPlaceFiringRate / I0 * 0.001
    def get_firings(self, t):
        ampl4gauss = 2 * (self.InPlacePeakRate - self.OutPlaceFiringRate) / (self.OutPlaceFiringRate + 1) #  range [-1, inf]

        multip = (1 + ampl4gauss * exp(-0.5 * ((t - self.CenterPlaceField) / self.SigmaPlaceField) ** 2))

        start_place = t - self.CenterPlaceField - 3 * self.SigmaPlaceField
        end_place = t - self.CenterPlaceField + 3 * self.SigmaPlaceField
        inplace = 0.25 * (1.0 - (start_place / (self.ALPHA + np.abs(start_place)))) * (
                1.0 + end_place / (self.ALPHA + abs(end_place)))

        precession = self.SlopePhasePrecession * inplace
        phases = self.OutPlaceThetaPhase * (1 - inplace) + self.PrecessionOnset * inplace #!!!!

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
class VonMisesLoss(tf.Module):
    def __init__(self, params, mask=None):
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

    def precomute(self):
        self.kappa = self.r2kappa(self.R)

        self.mult4time = tf.constant(2 * PI * self.ThetaFreq * 0.001, dtype=tf.float64)

        I0 = bessel_i0(self.kappa)
        self.normalizator = self.OutPlaceFiringRate / I0 * 0.001

    def get_firings(self, t):
        firings = self.normalizator * exp(self.kappa * cos(self.mult4time * t - self.phase) )
        return firings
    def get_loss(self, simulated_firings, t):
        t = tf.reshape(t, shape=(-1, 1))
        target_firings = self.get_firings(t)
        target_firings = tf.transpose(target_firings)
        selected_firings = tf.boolean_mask(simulated_firings, self.mask, axis=0)


        loss = tf.reduce_sum( target_firings * selected_firings )
        return loss
#########################################################################
default_params = {
            "R": 0.25,
            "OutPlaceFiringRate" : 0.5,
            "OutPlaceThetaPhase" : 3.14,
            "InPlacePeakRate" : 8.0,
            "CenterPlaceField" : 5000.0,
            "SigmaPlaceField" : 500,
            "SlopePhasePrecession" : np.deg2rad(10)*10 * 0.001,
            "PrecessionOnset" : -1.57,
            "ThetaFreq" : 8.0,
}

params = []

mask = tf.constant( [True, True, False, False, True], dtype=tf.dtypes.bool)

for _ in range(tf.reduce_sum(tf.cast(mask, dtype=tf.dtypes.int16))):
    params.append(default_params)


t = tf.range(0, 10000.0, 0.1, dtype=tf.float64)

genrators = SpatialThetaGenerators(params, mask)
genrators.precomute()

simulated_firings = tf.random.uniform( shape=(5, tf.size(t)), maxval=1.0, dtype=tf.dtypes.float64)
loss = genrators.get_loss(simulated_firings, t)

print(loss)

