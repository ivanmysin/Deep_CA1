import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
from genloss import WeightedMSE, WeightedLMSE

mask = np.ones(5, dtype='bool')
#mask[2:-1] = 0

y_true = tf.ones(shape=(1, 10, 5 ), dtype='float32' )
y_pred = tf.ones(shape=(1, 10, 5 ), dtype='float32' ) + 0.2

Loss_Func = WeightedLMSE(mask)


l = Loss_Func(y_true, y_pred)

print(l)

