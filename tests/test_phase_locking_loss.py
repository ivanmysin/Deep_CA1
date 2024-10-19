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

params = [
    { "R" : 0.5,
      "freq" : 8.0,
      "phase" : 3.14,
      "mean_spike_rate" : 1.5,
    },
]
genrators = VonMissesGenerators( params )
t = tf.range(0, 10000.0, 0.1, dtype=tf.float64)
theta_phases = 2 * np.pi * 0.001 * t * params[0]["freq"]
real = cos(theta_phases)
imag = sin(theta_phases)
print(genrators.trainable_variables)

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(genrators.trainable_variables)

    firings = genrators(t)
    firings_normal = firings / tf.math.reduce_sum(firings) # tf.nn.softmax(firings) #

    Rsim = sqrt( tf.math.reduce_sum( firings_normal * real )**2 + tf.math.reduce_sum(firings_normal * imag)**2 )

    loss = (Rsim - 0.6)**2
    grad = tape.gradient(loss, genrators.trainable_variables)
    print(grad)

sine = 0.5 * (np.cos(2 * np.pi * 0.001 * t * 8.0) + 1)



fig, ax= plt.subplots()
ax.plot(t, firings_normal, linewidth=4, color='green')
sine_ampls = sine * np.max(firings)
ax.plot(t, sine_ampls, linestyle="--", linewidth=1, color='black')
plt.show()

