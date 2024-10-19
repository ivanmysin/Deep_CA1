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
    def __call__(self, t):
        ## firings = self.normalizator * exp(self.kappa * cos(self.mult4time * t - self.phase) )


        # meanSR = params['mean_firing_rate']
        # phase = np.deg2rad(params['phase_out_place'])
        # kappa = r2kappa(params["R_place_cell"])
        # maxFiring = params['peak_firing_rate']
        #
        # SLOPE = np.deg2rad(params['precession_slope'] * v_an * 0.001)  # rad / ms
        # ONSET = np.deg2rad(params['precession_onset'])
        #
        # sigma_spt = params['sigma_place_field'] / v_an * 1000
        #
        # mult4time = 2 * np.pi * theta_freq * 0.001
        #
        # I0 = bessel_i0(kappa)
        # normalizator = meanSR / I0 * 0.001 * dt

        ampl4gauss = 2 * (self.InPlacePeakRate - self.OutPlaceFiringRate) / (self.OutPlaceFiringRate + 1) #  range [-1, inf]

        multip = (1 + ampl4gauss * exp(-0.5 * ((t - self.CenterPlaceField) / self.SigmaPlaceField) ** 2))

        start_place = t - self.CenterPlaceField - 3 * self.SigmaPlaceField
        end_place = t - self.CenterPlaceField + 3 * self.SigmaPlaceField
        inplace = 0.25 * (1.0 - (start_place / (self.ALPHA + np.abs(start_place)))) * (
                1.0 + end_place / (self.ALPHA + abs(end_place)))

        precession = self.SlopePhasePrecession * inplace
        phases = self.OutPlaceThetaPhase * (1 - inplace) + self.PrecessionOnset * inplace

        firings = self.normalizator * exp(self.kappa * cos((self.mult4time + precession) * t - phases))

        firings = multip * firings  # / (0.001 * dt)

        return firings


if __name__ == "__main__":
    params = [
        {
            "R": 0.25,
            "OutPlaceFiringRate" : 0.5,
            "OutPlaceThetaPhase" : 3.14,
            "InPlacePeakRate" : 8.0,
            "CenterPlaceField" : 5000.0,
            "SigmaPlaceField" : 500,
            "SlopePhasePrecession" : np.deg2rad(10)*10 * 0.001,
            "PrecessionOnset" : -1.57,
            "ThetaFreq" : 8.0,
         },
    ]
    genrators = SpatialThetaGenerators(params)

    t = tf.range(0, 10000.0, 0.1, dtype=tf.float64)
    theta_phases = 2 * np.pi * 0.001 * t * params[0]["ThetaFreq"]
    real = cos(theta_phases)
    imag = sin(theta_phases)
    print(genrators.trainable_variables)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(genrators.trainable_variables)
        genrators.precomute()
        firings = genrators(t)
        firings_normal = firings / tf.math.reduce_sum(firings)  # tf.nn.softmax(firings) #

        Rsim = sqrt(tf.math.reduce_sum(firings_normal * real) ** 2 + tf.math.reduce_sum(firings_normal * imag) ** 2)

        print(Rsim)
        loss = (Rsim - 0.6) ** 2
        grad = tape.gradient(loss, genrators.trainable_variables)
        print(grad)

    sine = 0.5 * (np.cos(2 * np.pi * 0.001 * t * 8.0) + 1)

    fig, ax = plt.subplots()
    ax.plot(t, firings, linewidth=4, color='green')
    sine_ampls = sine * np.max(firings)
    ax.plot(t, sine_ampls, linestyle="--", linewidth=1, color='black')
    plt.show()