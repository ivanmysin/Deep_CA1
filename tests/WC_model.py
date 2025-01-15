import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, RNN
import sys
sys.path.append('../')
import myconfig

class CW_layer(Layer):

    def __init__(self, params, dt=0.1, **kwargs):
        super(CW_layer, self).__init__(**kwargs)
        self.dt = dt
        self.r =  self.add_weight(shape = K.shape(params['r']),
                                        initializer = keras.initializers.Constant(params['r']),
                                        trainable = True,
                                        dtype = myconfig.DTYPE,
                                        constraint = keras.constraints.NonNeg(),
                                        name = f"r")


        self.tau_nu = self.add_weight(shape = K.shape(params['tau_nu']),
                                        initializer = keras.initializers.Constant(params['tau_nu']),
                                        trainable = True,
                                        dtype = myconfig.DTYPE,
                                        constraint = keras.constraints.NonNeg(),
                                        name = f"tau_nu")

        self.M = self.add_weight(shape = K.shape(params['M']),
                                        initializer = keras.initializers.Constant(params['M']),
                                        trainable = True,
                                        dtype = myconfig.DTYPE,
                                        constraint = keras.constraints.NonNeg(),
                                        name = f"M")

        self.sigma = self.add_weight(shape = K.shape(params['sigma']),
                                        initializer = keras.initializers.Constant(params['sigma']),
                                        trainable = True,
                                        dtype = myconfig.DTYPE,
                                        constraint = keras.constraints.NonNeg(),
                                        name = f"sigma")

        self.th = self.add_weight(shape = K.shape(params['th']),
                                        initializer = keras.initializers.Constant(params['th']),
                                        trainable = True,
                                        dtype = myconfig.DTYPE,
                                        name = f"th")

        self.state_size = [1, ]

    def build(self, input_shape):
        super(CW_layer, self).build(input_shape)

        # self.ext_tau_nu = K.exp(-self.dt / self.tau_nu)
        # self.sigma = tf.convert_to_tensor(params['sigma'], dtype=myconfig.DTYPE)

        self.built = True


    def Phi(self, x):
        y = self.M * x / (x**2 + self.sigma**2)
        y = y * tf.nn.relu(x + self.th)

    def call(self, inputs, states):
        E = inputs
        nu_prev = states




        return output, [R, U, A, Vsyn]

    def get_initial_state(self, batch_size=None):
        initial_state = K.zeros(batch_size, 1)