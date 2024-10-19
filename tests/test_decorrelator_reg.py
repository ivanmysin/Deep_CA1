import os
os.chdir("../")
import numpy as np
from scipy.stats import zscore
from genloss import Decorrelator
import matplotlib.pyplot as plt


X = np.random.uniform(0, 10, 2000).reshape(500, 4)

t = np.linspace(0, 1, 500)

X[:, 0] = np.cos(2*np.pi*t*5)
X[:, 1] = np.cos(2*np.pi*t*5 + np.pi)
X[:, 2] = np.cos(2*np.pi*t*5 + 0.5*np.pi)
X[:, 2] = np.cos(2*np.pi*t*7 + 0.5*np.pi)



"""
def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    cov_xx = tf.matmul(tf.transpose(x-mean_x), x-mean_x)/tf.cast(tf.shape(x)[0]-1, tf.float64)
    return cov_xx
"""

frings = X.reshape(1, 500, 4 ).astype(np.float32)

decor = Decorrelator(strength=1.0)
loss = decor(frings)

print(loss)

#cov_matrix = np.cov(X)
cov_matrix = np.corrcoef(X.T)


Xcentered = X - np.mean(X, axis=0, keepdims=True)
Xcentered = Xcentered / np.sqrt( np.mean(Xcentered**2, axis=0, keepdims=True) )

mycorr = np.dot( Xcentered.T, Xcentered) / Xcentered.shape[0]

# print(mycorr)
# print("#############################################")
#
#
# print(cov_matrix)

# print(  np.sum(  (mycorr - cov_matrix)**2  ) )
print(  np.mean(  cov_matrix**2  ) )
print(  np.mean(  mycorr**2  ) )


