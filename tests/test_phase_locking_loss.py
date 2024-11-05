import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


erf = tf.math.erf
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

SQRT_FROM_2 = np.sqrt(2)
SQRT_FROM_2_PI = 0.7978845608028654
PI = np.pi


class SpatialThetaGenerators(tf.Module):
    def __init__(self, params):
        super(SpatialThetaGenerators, self).__init__()
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


        self.ThetaFreq = tf.Variable(ThetaFreq, dtype=tf.float64, name="Theta_freq")
        self.OutPlaceFiringRate = tf.Variable(OutPlaceFiringRate, dtype=tf.float64, name="OutPlaceFiringRate")
        self.OutPlaceThetaPhase = tf.Variable(OutPlaceThetaPhase, dtype=tf.float64, name="OutPlaceThetaPhase")
        self.R = tf.Variable(Rs, dtype=tf.float64, name="R")
        self.InPlacePeakRate = tf.Variable(InPlacePeakRate, dtype=tf.float64, name="InPlacePeakRate")
        self.CenterPlaceField = tf.Variable(CenterPlaceField, dtype=tf.float64, name="CenterPlaceField")
        self.SigmaPlaceField = tf.Variable(SigmaPlaceField, dtype=tf.float64, name="SigmaPlaceField")
        self.SlopePhasePrecession = tf.Variable(SlopePhasePrecession, dtype=tf.float64, name="SlopePhasePrecession")
        self.PrecessionOnset = tf.Variable(PrecessionOnset, dtype=tf.float64, name="PrecessionOnset") #, [1, -1])

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

    def build(self):
        input_shape = (None, 1)

        #super(SpatialThetaGenerators, self).build(input_shape)

        self.kappa = self.r2kappa(self.R)

        tmp = 2 * PI * self.ThetaFreq * 0.001
        self.mult4time = tf.constant(tmp, dtype=tf.float64)

        I0 = bessel_i0(self.kappa)
        self.normalizator = self.OutPlaceFiringRate / I0


        self.built = True


    def __call__(self, t):
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
        #firings = tf.reshape(firings, shape=(1, tf.shape(firings)[0], tf.shape(firings)[1]))

        return firings

class VonMissesGenerators(tf.Module):

    def __init__(self, params, dt=0.1, start_idx=0):
        super(VonMissesGenerators, self).__init__()

        Rs = []
        omegas = []
        phases = []
        mean_spike_rates = []


        for params_el in params:
            if "target" in params_el.keys():
                p = params_el["target"]
            else:
                p = params_el

            Rs.append(p["R"])
            omegas.append( p["freq"] )
            phases.append( p["phase"] )
            mean_spike_rates.append(p["mean_spike_rate"] )

        self.omega = tf.constant(omegas, dtype=tf.float64)
        self.phase = tf.Variable(phases, dtype=tf.float64, name="Theta_phase")
        self.mean_spike_rate = tf.Variable(mean_spike_rates, dtype=tf.float64, name="Mean_firing_rate")

        self.R = tf.Variable(Rs, dtype=tf.float64, name="Ray_length")



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
        kappa = tf.where(R < 0.53,  2 * R + R ** 3 + 5 / 6 * R ** 5, 0.0)
        kappa = tf.where(logical_and(R >= 0.53, R < 0.85),  -0.4 + 1.39 * R + 0.43 / (1 - R), kappa)
        kappa = tf.where(R >= 0.85,  1 / (3 * R - 4 * R ** 2 + R ** 3), kappa)
        return kappa

    def __call__(self, t):
        self.kappa = self.r2kappa(self.R)

        self.mult4time = tf.constant(2 * PI * self.omega * 0.001, dtype=tf.float64)

        I0 = bessel_i0(self.kappa)
        self.normalizator = self.mean_spike_rate / I0 * 0.001

        firings = self.normalizator * exp(self.kappa * cos(self.mult4time * t - self.phase) )
        return firings

###########################################################

# params = [
#     { "R" : 0.5,
#       "freq" : 8.0,
#       "phase" : 3.14,
#       "mean_spike_rate" : 1.5,
#     },
# ]
# genrators = VonMissesGenerators( params )

params = [
    {
        "ThetaFreq" : 7,
        "OutPlaceFiringRate" : 1.5,
        "OutPlaceThetaPhase" : 0,
        "InPlacePeakRate" : 8.0,
        "CenterPlaceField" : 140,
        "R" : 0.01,
        "SigmaPlaceField" : 500,
        "SlopePhasePrecession" : 0.0,
        "PrecessionOnset" : 0.0,
    },
]

genrators = SpatialThetaGenerators( params )


ThetaFreq = params[0]["ThetaFreq"]

print(genrators.trainable_variables)

t = tf.range(0, 140.0, 0.1, dtype=tf.float64)
theta_phases = 2 * np.pi * 0.001 * t * 8.0
real = cos(theta_phases)
imag = sin(theta_phases)


phi = tf.constant(params[0]["OutPlaceThetaPhase"]+1.5, dtype=tf.float64)

imagtrue = tf.constant( (params[0]["R"]+0.3) * sin(phi), dtype=tf.float64 )
realtrue = tf.constant( (params[0]["R"]+0.3) * cos(phi), dtype=tf.float64 )

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(genrators.trainable_variables)

    genrators.build()

    firings = genrators(t)
    firings = tf.reshape(firings, shape=(-1, ))

    firings_normal = firings / tf.math.reduce_sum(firings) # tf.nn.softmax(firings) #

    realsim = firings_normal * real
    imagsim = firings_normal * imag
    Rsim = sqrt( tf.math.reduce_sum( realsim )**2 + tf.math.reduce_sum(imagsim)**2 )

    #loss = (Rsim - 0.6)**2

    loss = (tf.reduce_sum(imagsim) - imagtrue)**2 + (tf.reduce_sum(realsim) - realtrue)**2

    grad = tape.gradient(loss, genrators.trainable_variables)

    for g, tv in zip(grad, genrators.trainable_variables):
        print(tv.name, g)

sine = 0.5 * (np.cos(2 * np.pi * 0.001 * t * ThetaFreq) + 1)



fig, ax = plt.subplots()
ax.plot(t, firings_normal, linewidth=4, color='green')
sine_ampls = sine * np.max(firings_normal)
ax.plot(t, sine_ampls, linestyle="--", linewidth=1, color='black')
plt.show()

