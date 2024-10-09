import os

import numpy as np

os.chdir("../")

import tensorflow as tf
import genloss


mask = np.random.uniform(0, 1, 10) > 0.5
input = 1000 * np.random.uniform(0, 10, 100).reshape(1, 10, 10)


regularizer = genloss.RobastMeanOutRanger()

layer = genloss.FrequencyFilter(mask)
layer.activity_regularizer = regularizer
out = layer(input)
print(out.shape)

print( layer.losses )

