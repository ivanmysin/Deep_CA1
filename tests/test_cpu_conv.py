import tensorflow as tf
import numpy as np
from tensorflow import GradientTape


x = tf.Variable(np.ones(3), dtype=tf.complex64)
kernel = tf.constant( np.ones(3), dtype=tf.complex64)


with GradientTape() as tape:
    Nfull = tf.size(kernel) + tf.size(x) - 1



    K = tf.signal.fft(kernel)
    X = tf.signal.fft(x)

    C = tf.signal.ifft(K * X)

    C = tf.math.real(C)

    Cs = tf.reduce_mean(C)

grad = tf.cast( tape.gradient(Cs, x), dtype=tf.float32)

print(grad)