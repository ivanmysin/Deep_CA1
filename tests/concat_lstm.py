import numpy as np
import tensorflow as tf

x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
concatted = tf.keras.layers.Concatenate()([x1, x2])
print(concatted.shape)