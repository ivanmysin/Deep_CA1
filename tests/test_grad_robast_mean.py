import tensorflow as tf
import numpy as np
from tensorflow import GradientTape

x = tf.Variable(np.zeros(10), dtype=tf.float32)
ThetaFreq = 8

with GradientTape() as tape:
    #y = tf.math.exp( tf.reduce_mean (tf.math.log(x) )  )
    t_max = tf.cast(tf.size(x), dtype=tf.float32) * 0.1
    t = tf.range(0, t_max, 0.1)
    #t = tf.reshape(t, shape=(-1, 1))

    theta_phases = 2 * np.pi * 0.001 * t * ThetaFreq
    real = tf.math.cos(theta_phases)
    imag = tf.math.sin(theta_phases)

    normed_firings = x / (tf.reduce_sum(x) + 0.00000000001)
    #y = tf.math.sqrt(tf.reduce_sum(normed_firings * real) ** 2 + tf.reduce_sum(normed_firings * imag) ** 2)
    y = tf.math.sqrt(tf.reduce_sum(normed_firings * real) ** 2 + tf.reduce_sum(normed_firings * imag) ** 2 + 0.0000001)



print(tf.reduce_mean(x))
print(y)
g = tape.gradient(y, x)
print(g)