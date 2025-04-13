#import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN
from tensorflow.keras.constraints import Constraint
import izhs_lib

import sys
sys.path.append('../')
import myconfig


PI = 3.141592653589793
exp = tf.math.exp
tf.keras.backend.set_floatx(myconfig.DTYPE)


class ZeroOnesWeights(Constraint):
    """Ограничивает веса модели значениями между 0 и 1."""

    def __call__(self, w):
        return tf.clip_by_value(w, clip_value_min=0, clip_value_max=1)

class MeanFieldNetwork(Layer):

    def __init__(self, params, dt_dim, **kwargs):
        super().__init__(**kwargs)

        self.dt_dim = dt_dim

        self.units = len(params['alpha'])
        self.alpha = tf.convert_to_tensor( params['alpha'] )
        self.a = tf.convert_to_tensor( params['a'] )
        self.b = tf.convert_to_tensor( params['b'] )
        self.w_jump = tf.convert_to_tensor( params['w_jump'] )
        self.dts_non_dim = tf.convert_to_tensor( params['dts_non_dim'] )
        self.Delta_eta = tf.convert_to_tensor( params['Delta_eta'])
        # bar_eta = params['bar_eta']
        I_ext = tf.convert_to_tensor( params['I_ext'] )
        self.I_ext = self.add_weight(shape=tf.keras.ops.shape(I_ext),
                                        initializer=tf.keras.initializers.Constant(I_ext),
                                        trainable=False,
                                        dtype=myconfig.DTYPE,
                                        constraint=tf.keras.constraints.NonNeg(),
                                        name=f"I_ext")

        gsyn_max = tf.convert_to_tensor(params['gsyn_max'])
        tau_f = tf.convert_to_tensor(params['tau_f'])
        tau_d = tf.convert_to_tensor(params['tau_d'])
        tau_r = tf.convert_to_tensor(params['tau_r'])
        Uinc = tf.convert_to_tensor(params['Uinc'])

        self.e_r = tf.convert_to_tensor(params['e_r'])
        self.pconn = tf.convert_to_tensor(params['pconn'])


        self.gsyn_max = self.add_weight(shape=tf.keras.ops.shape(gsyn_max),
                                        initializer=tf.keras.initializers.Constant(gsyn_max),
                                        # regularizer=self.gmax_regulizer,  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        trainable=False,
                                        dtype=myconfig.DTYPE,
                                        constraint=tf.keras.constraints.NonNeg(),
                                        name=f"gsyn_max")

        self.tau_f = self.add_weight(shape=tf.keras.ops.shape(tau_f),
                                     initializer=tf.keras.initializers.Constant(tau_f),
                                     trainable=False,
                                     dtype=myconfig.DTYPE,
                                     constraint=tf.keras.constraints.NonNeg(),
                                     name=f"tau_f")

        self.tau_d = self.add_weight(shape=tf.keras.ops.shape(tau_d),
                                     initializer=tf.keras.initializers.Constant(tau_d),
                                     trainable=False,
                                     dtype=myconfig.DTYPE,
                                     constraint=tf.keras.constraints.NonNeg(),
                                     name=f"tau_d_{self.pop_idx}")

        self.tau_r = self.add_weight(shape=tf.keras.ops.shape(tau_r),
                                     initializer=tf.keras.initializers.Constant(tau_r),
                                     trainable=False,
                                     dtype=myconfig.DTYPE,
                                     constraint=tf.keras.constraints.NonNeg(),
                                     name=f"tau_r_{self.pop_idx}")

        self.Uinc = self.add_weight(shape=tf.keras.ops.shape(Uinc),
                                    initializer=tf.keras.initializers.Constant(Uinc),
                                    trainable=False,
                                    dtype=myconfig.DTYPE,
                                    constraint=ZeroOnesWeights(),
                                    name=f"Uinc")

        #### !!!!!!!!!!!!!
        self.state_size = [self.units, self.units, self.units, (self.units, self.units), (self.units, self.units), (self.units, self.units)]

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def get_initial_state(self, batch_size=1):
        shape = [self.units, self.units]

        r = tf.zeros( [1, self.units], dtype=myconfig.DTYPE)
        v = tf.ones( [1, self.units], dtype=myconfig.DTYPE)
        w = tf.zeros( [1, self.units], dtype=myconfig.DTYPE)

        R = tf.ones( shape, dtype=myconfig.DTYPE)
        U = tf.zeros( shape, dtype=myconfig.DTYPE)
        A = tf.zeros( shape, dtype=myconfig.DTYPE)

        initial_state = [r, v, w, R, U, A]

        return initial_state

    def call(self, inputs, states):
        rates = states[0]
        v_avg = states[1]
        w_avg = states[2]
        R = states[3]
        U = states[4]
        A = states[5]

        g_syn = self.gsyn_max * A
        g_syn_tot = tf.math.reduce_sum(g_syn, axis=0)


        Isyn = tf.math.reduce_sum(g_syn * (self.e_r - v_avg), axis=0)

        rates = rates + self.dts_non_dim * (self.Delta_eta / PI + 2 * rates * v_avg - (self.alpha + g_syn_tot) * rates)
        v_avg = v_avg + self.dts_non_dim * (
                    v_avg ** 2 - self.alpha * v_avg - w_avg + self.I_ext + Isyn - (PI*rates)**2)
        w_avg = w_avg + self.dts_non_dim  * (a * (b * v_avg - w_avg) + self.w_jump * rates)

        FRpre_normed = self.pconn * self.dts_non_dim * rates
        FRpre_normed = tf.reshape(FRpre_normed, shape=(-1, 1))

        tau1r = tf.where(self.tau_d != self.tau_r, self.tau_d / (self.tau_d - self.tau_r), 1e-13)


        exp_tau_d = exp(-self.dt_dim / self.tau_d)
        exp_tau_f = exp(-self.dt_dim / self.tau_f)
        exp_tau_r = exp(-self.dt_dim / self.tau_r)


        a_ = A * exp_tau_d
        r_ = 1 + (R - 1 + tau1r * A) * exp_tau_r  - tau1r * A
        u_ = U * exp_tau_f

        U = u_ + self.Uinc * (1 - u_) * FRpre_normed
        A = a_ + U * r_ * FRpre_normed
        R = r_ - U * r_ * FRpre_normed


        output = rates

        return output, [rates, v_avg, w_avg, R, U, A]


if __name__ == '__main__':

    pass