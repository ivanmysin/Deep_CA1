import numpy as np
import tensorflow as tf
from tensorflow.keras import ops
from tensorflow.keras.layers import Layer, RNN, Input
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.regularizers import Regularizer, L2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model

import izhs_lib
import h5py

import sys


sys.path.append('../')
import myconfig

PI = pi = tf.constant(3.141592653589793, dtype=myconfig.DTYPE)
exp = tf.math.exp
tf.keras.backend.set_floatx(myconfig.DTYPE)


#@tf.keras.utils.register_keras_serializable(package="SaveFirings")
class SaveFirings(Callback):
    def __init__(self, firing_model, t_full, path, filename_template, save_freq=4, **kwargs):
        super().__init__(**kwargs)

        self.firing_model = firing_model
        self.t_full = t_full
        self.path = path
        self.filename_template = filename_template
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if ( (epoch+1) % self.save_freq) != 0:
            return

        filepath = self.path + self.filename_template.format(epoch=epoch)
        firings = self.firing_model.predict(self.t_full)

        with h5py.File(filepath, mode='w') as h5file:
            h5file.create_dataset('firings', data=firings)

@tf.keras.utils.register_keras_serializable(package="ZeroWallReg")
class ZeroWallReg(Regularizer):

    def __init__(self, lw=0.01, close_coeff=100, eps=0.001):
        self.close_coeff = close_coeff
        self.lw = lw
        self.eps = eps

    def __call__(self, x):
        return tf.reduce_sum(-self.lw * tf.math.log( (x * self.close_coeff) + self.eps ) )

    def get_config(self):
        config = {
            "close_coeff" : float(self.close_coeff),
            "lw" : float(self.lw),
            "eps" : float(self.eps),

        }

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class BoundWallReg(Regularizer):
    def __init__(self, min_val=None, max_val=None, lw=0.01, close_coeff=100, eps=0.001):
        self.close_coeff = close_coeff
        self.lw = lw
        self.eps = eps

        # Если границы не заданы, используем стандартные 0 и 1
        if min_val is None:
            self.min_val = tf.constant(0.0, dtype=myconfig.DTYPE)
        else:
            self.min_val = tf.constant(min_val, dtype=myconfig.DTYPE)

        if max_val is None:
            self.max_val = tf.constant(1.0, dtype=myconfig.DTYPE)
        else:
            self.max_val = tf.constant(max_val, dtype=myconfig.DTYPE)

    def __call__(self, x):
        # Нормализуем x к диапазону [0, 1]
        x_normalized = (x - self.min_val) / (self.max_val - self.min_val + self.eps)

        x_normalized = tf.clip_by_value(x_normalized, self.eps, 1.0 - self.eps)

        # Применяем регуляризацию как в исходном коде, но к нормализованному тензору
        reg_term = -self.lw * (
                tf.reduce_sum(tf.math.log(x_normalized * self.close_coeff)) +
                tf.reduce_sum(tf.math.log((1.0 - x_normalized) * self.close_coeff))
        )
        return reg_term

    def get_config(self):
        # Пытаемся преобразовать тензоры в сериализуемый формат
        try:
            min_val_np = self.min_val.numpy() if hasattr(self.min_val, 'numpy') else self.min_val
            max_val_np = self.max_val.numpy() if hasattr(self.max_val, 'numpy') else self.max_val
        except:
            min_val_np = self.min_val
            max_val_np = self.max_val

        config = {
            "min_val": min_val_np,
            "max_val": max_val_np,
            "close_coeff": float(self.close_coeff),
            "lw": float(self.lw),
            "eps": float(self.eps),
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.keras.utils.register_keras_serializable(package="ZeroOneWallReg")
class ZeroOneWallReg(ZeroWallReg):
    def __call__(self, x):
        res = -self.lw * (tf.reduce_sum( tf.math.log(x * self.close_coeff)) + tf.reduce_sum( tf.math.log( (1.0 - x) * self.close_coeff)))
        return res



@tf.keras.utils.register_keras_serializable(package="MinMaxWeights")
class MinMaxWeights(Constraint):

    def __init__(self, min_val=0, max_val=10000):
        self.min = tf.convert_to_tensor(min_val) if not isinstance(min_val, tf.Tensor) else min_val
        self.max = tf.convert_to_tensor(max_val) if not isinstance(max_val, tf.Tensor) else max_val


    def __call__(self, w):
        return tf.clip_by_value(w, clip_value_min=self.min, clip_value_max=self.max)

    def get_config(self):
        config = {
            "min": self.min.numpy().tolist() if hasattr(self.min, "numpy") else self.min.tolist(),
            "max": self.max.numpy().tolist() if hasattr(self.max, "numpy") else self.max.tolist(),
        }

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.keras.utils.register_keras_serializable(package="MeanFieldNetwork")
class MeanFieldNetwork(Layer):

    def __init__(self, params, dt_dim=0.5, use_input=False, stability_penalty=1e-3, **kwargs):
        super().__init__(**kwargs)

        self.dt_dim = dt_dim
        self.use_input = use_input

        self.stability_penalty = stability_penalty

        self.units = len(params['alpha'])
        self.alpha = tf.convert_to_tensor( params['alpha'], dtype=myconfig.DTYPE )
        self.a = tf.convert_to_tensor( params['a'], dtype=myconfig.DTYPE )
        self.b = tf.convert_to_tensor( params['b'], dtype=myconfig.DTYPE )
        self.w_jump = tf.convert_to_tensor( params['w_jump'], dtype=myconfig.DTYPE )
        self.dts_non_dim = tf.convert_to_tensor( params['dts_non_dim'], dtype=myconfig.DTYPE )

        Delta_eta = tf.convert_to_tensor(params['Delta_eta'], dtype=myconfig.DTYPE)
        Delta_eta_min = tf.convert_to_tensor(params['Delta_eta_min'], dtype=myconfig.DTYPE)
        Delta_eta_max = tf.convert_to_tensor(params['Delta_eta_max'], dtype=myconfig.DTYPE)

        self.Delta_eta = self.add_weight(shape=tf.keras.ops.shape(Delta_eta),
                                        initializer=tf.keras.initializers.Constant(Delta_eta),
                                        trainable=True,
                                        dtype=myconfig.DTYPE,
                                        constraint=MinMaxWeights(min_val=Delta_eta_min, max_val=Delta_eta_max),
                                        name=f"Delta_eta")

        I_ext = tf.convert_to_tensor( params['I_ext'], dtype=myconfig.DTYPE )
        self.I_ext = self.add_weight(shape=tf.keras.ops.shape(I_ext),
                                        initializer=tf.keras.initializers.Constant(I_ext),
                                        trainable=True,
                                        # regularizer=L2(l2=25.0), #!!!!!!!!!!!!!!!!!
                                        regularizer=L2(l2=10.0),
                                        dtype=myconfig.DTYPE,
                                        name=f"I_ext")

        gsyn_max = tf.convert_to_tensor(params['gsyn_max'], dtype=myconfig.DTYPE)
        tau_f = tf.convert_to_tensor(params['tau_f'], dtype=myconfig.DTYPE)
        tau_d = tf.convert_to_tensor(params['tau_d'], dtype=myconfig.DTYPE)
        tau_r = tf.convert_to_tensor(params['tau_r'], dtype=myconfig.DTYPE)
        Uinc = tf.convert_to_tensor(params['Uinc'], dtype=myconfig.DTYPE)

        self.e_r = tf.convert_to_tensor(params['e_r'], dtype=myconfig.DTYPE)
        pconn = tf.convert_to_tensor(params['pconn'])

        self.pconn = self.add_weight(shape=tf.keras.ops.shape(pconn),
                                        initializer=tf.keras.initializers.Constant(pconn),
                                        # regularizer=self.gmax_regulizer,  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        trainable=False,
                                        dtype=myconfig.DTYPE,
                                        constraint=MinMaxWeights(min_val=0, max_val=1),
                                        name=f"pconn")


        self.gsyn_max = self.add_weight(shape=tf.keras.ops.shape(gsyn_max),
                                        initializer = tf.keras.initializers.Constant(gsyn_max),
                                        regularizer = L2(l2=0.0001),  #
                                        # regularizer=ZeroWallReg(lw=0.00001, close_coeff=100000),
                                        trainable=True,
                                        dtype=myconfig.DTYPE,
                                        constraint=tf.keras.constraints.NonNeg(),
                                        name=f"gsyn_max")

        self.tau_f = self.add_weight(shape=tf.keras.ops.shape(tau_f),
                                     initializer=tf.keras.initializers.Constant(tau_f),
                                     # regularizer=ZeroWallReg(lw=0.001, close_coeff=1000),
                                     trainable=False,
                                     dtype=myconfig.DTYPE,
                                     constraint=MinMaxWeights(min_val=6.0, max_val=240.0),
                                     name=f"tau_f")

        self.tau_d = self.add_weight(shape=tf.keras.ops.shape(tau_d),
                                     initializer=tf.keras.initializers.Constant(tau_d),
                                     # regularizer=ZeroWallReg(lw=0.001, close_coeff=1000),
                                     trainable=False,
                                     dtype=myconfig.DTYPE,
                                     constraint=MinMaxWeights(min_val=2.0, max_val=15.0),
                                     name=f"tau_d")

        self.tau_r = self.add_weight(shape=tf.keras.ops.shape(tau_r),
                                     initializer=tf.keras.initializers.Constant(tau_r),
                                     # regularizer=ZeroWallReg(lw=0.001, close_coeff=1000),
                                     trainable=False,
                                     dtype=myconfig.DTYPE,
                                     constraint=MinMaxWeights(min_val=91.0, max_val=1300.0),
                                     name=f"tau_r")

        self.Uinc = self.add_weight(shape=tf.keras.ops.shape(Uinc),
                                    initializer=tf.keras.initializers.Constant(Uinc),
                                    # regularizer=ZeroOneWallReg(lw=0.001, close_coeff=1000),
                                    trainable=False,
                                    dtype=myconfig.DTYPE,
                                    constraint=MinMaxWeights(min_val=0.04, max_val=0.7),
                                    name=f"Uinc")

        synaptic_matrix_shapes = tf.shape(self.gsyn_max)

        self.is_nmda = False
        if 'nmda' in params.keys():
            self.is_nmda = True

            self.pconn_nmda = tf.convert_to_tensor(params['nmda']['pconn_nmda'], dtype=myconfig.DTYPE)
            self.Mgb = tf.convert_to_tensor(params['nmda']['Mgb'], dtype=myconfig.DTYPE)
            self.av_nmda = tf.convert_to_tensor(params['nmda']['av_nmda'], dtype=myconfig.DTYPE)

            gsyn_max_nmda = tf.convert_to_tensor(params['nmda']['gsyn_max_nmda'], dtype=myconfig.DTYPE)
            self.gsyn_max_nmda = self.add_weight(shape=tf.keras.ops.shape(gsyn_max_nmda),
                                    initializer=tf.keras.initializers.Constant(gsyn_max_nmda),
                                    # regularizer=ZeroOneWallReg(lw=0.001, close_coeff=1000),
                                    trainable=True,
                                    dtype=myconfig.DTYPE,
                                    constraint=tf.keras.constraints.NonNeg(),
                                    name=f"gsyn_max_nmda")

            tau1_nmda = tf.convert_to_tensor(params['nmda']['tau1_nmda'], dtype=myconfig.DTYPE)

            self.tau1_nmda = self.add_weight(shape=tf.keras.ops.shape(tau1_nmda),
                                     initializer=tf.keras.initializers.Constant(tau1_nmda),
                                     # regularizer=ZeroWallReg(lw=0.001, close_coeff=1000),
                                     trainable=True,
                                     dtype=myconfig.DTYPE,
                                     constraint=MinMaxWeights(min_val=dt_dim),  #tf.keras.constraints.NonNeg(),
                                     name=f"tau1_nmda")

            tau2_nmda = tf.convert_to_tensor(params['nmda']['tau2_nmda'], dtype=myconfig.DTYPE)
            self.tau2_nmda = self.add_weight(shape=tf.keras.ops.shape(tau2_nmda),
                                     initializer=tf.keras.initializers.Constant(tau2_nmda),
                                     # regularizer=ZeroWallReg(lw=0.001, close_coeff=1000),
                                     trainable=True,
                                     dtype=myconfig.DTYPE,
                                     constraint=MinMaxWeights(min_val=dt_dim),  #tf.keras.constraints.NonNeg(),
                                     name=f"tau2_nmda")

        if self.is_nmda:
            self.state_size = [self.units, self.units, self.units, synaptic_matrix_shapes, synaptic_matrix_shapes,
                           synaptic_matrix_shapes, synaptic_matrix_shapes, synaptic_matrix_shapes]
        else:
            self.state_size = [self.units, self.units, self.units, synaptic_matrix_shapes, synaptic_matrix_shapes, synaptic_matrix_shapes]

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def get_initial_state(self, batch_size=1):
        shape = [self.units+3, self.units]

        r = tf.zeros( [1, self.units], dtype=myconfig.DTYPE)
        v = tf.zeros( [1, self.units], dtype=myconfig.DTYPE)
        w = tf.zeros( [1, self.units], dtype=myconfig.DTYPE)

        synaptic_matrix_shapes = tf.shape(self.gsyn_max)

        R = tf.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)
        U = tf.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)
        A = tf.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)

        # error_estimate = tf.zeros( [1, 1], dtype=myconfig.DTYPE)

        if self.is_nmda:
            dgnmda = tf.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)
            gnmda = tf.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)

            initial_state = [r, v, w, R, U, A, gnmda, dgnmda]
        else:
            initial_state = [r, v, w, R, U, A]

        return initial_state

    def get_rate_derivative(self, rates, v_avg, g_syn_tot):
        drdt = self.Delta_eta / PI + 2 * rates * v_avg - (self.alpha + g_syn_tot) * rates
        return drdt

    def get_v_avg_derivative(self, rates, v_avg, w_avg, g_syn):
        Isyn = tf.reduce_sum(g_syn * (self.e_r - v_avg), axis=0)
        dvdt = v_avg ** 2 - self.alpha * v_avg - w_avg + self.I_ext + Isyn - (PI * rates)**2
        return dvdt

    def get_w_avg_derivative(self, rates, v_avg, w_avg):
        dwdt = self.a * (self.b * v_avg - w_avg) + self.w_jump * rates
        return dwdt

    def runge_kutta_step(self, rates, v_avg, w_avg, g_syn):
        g_syn_tot = tf.reduce_sum(g_syn, axis=0)

        # РК4 шаг
        k1_rates = self.get_rate_derivative(rates, v_avg, g_syn_tot)
        k1_v = self.get_v_avg_derivative(rates, v_avg, w_avg, g_syn)
        k1_w = self.get_w_avg_derivative(rates, v_avg, w_avg)

        half_rate = rates + 0.5 * self.dts_non_dim * k1_rates
        half_v = v_avg + 0.5 * self.dts_non_dim * k1_v
        half_w = w_avg + 0.5 * self.dts_non_dim * k1_w

        k2_rates = self.get_rate_derivative(half_rate, half_v, g_syn_tot)
        k2_v = self.get_v_avg_derivative(half_rate, half_v, half_w, g_syn)
        k2_w = self.get_w_avg_derivative(half_rate, half_v, half_w)

        half_rate = rates + 0.5 * self.dts_non_dim * k2_rates
        half_v = v_avg + 0.5 * self.dts_non_dim * k2_v
        half_w = w_avg + 0.5 * self.dts_non_dim * k2_w

        k3_rates = self.get_rate_derivative(half_rate, half_v, g_syn_tot)
        k3_v = self.get_v_avg_derivative(half_rate, half_v, half_w, g_syn)
        k3_w = self.get_w_avg_derivative(half_rate, half_v, half_w)

        half_rate = rates + self.dts_non_dim * k3_rates
        half_v = v_avg + self.dts_non_dim * k3_v
        half_w = w_avg + self.dts_non_dim * k3_w

        k4_rates = self.get_rate_derivative(half_rate, half_v, g_syn_tot)
        k4_v = self.get_v_avg_derivative(half_rate, half_v, half_w, g_syn)
        k4_w = self.get_w_avg_derivative(half_rate, half_v, half_w)

        rates_rk4 = rates + self.dts_non_dim * (k1_rates + 2 * k2_rates + 2 * k3_rates + k4_rates) / 6.0
        v_avg_rk4 = v_avg + self.dts_non_dim * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
        w_avg_rk4 = w_avg + self.dts_non_dim * (k1_w + 2 * k2_w + 2 * k3_w + k4_w) / 6.0

        # РК2 шаг для оценки ошибки
        rates_rk2_k1 = self.dts_non_dim * k1_rates
        v_avg_rk2_k1 = self.dts_non_dim * k1_v
        w_avg_rk2_k1 = self.dts_non_dim * k1_w

        rates_rk2_k2 = self.dts_non_dim * self.get_rate_derivative(rates + rates_rk2_k1, v_avg + v_avg_rk2_k1, g_syn_tot)
        v_avg_rk2_k2 = self.dts_non_dim * self.get_v_avg_derivative(rates + rates_rk2_k1, v_avg + v_avg_rk2_k1, w_avg + w_avg_rk2_k1, g_syn)
        w_avg_rk2_k2 = self.dts_non_dim * self.get_w_avg_derivative(rates + rates_rk2_k1, v_avg + v_avg_rk2_k1, w_avg + w_avg_rk2_k1)

        rates_rk2 = rates + 0.5 * (rates_rk2_k1 + rates_rk2_k2)
        v_avg_rk2 = v_avg + 0.5 * (v_avg_rk2_k1 + v_avg_rk2_k2)
        w_avg_rk2 = w_avg + 0.5 * (w_avg_rk2_k1 + w_avg_rk2_k2)

        # # Оценка локальной ошибки
        # if self.stability_penalty > 0.0:
        #     error_estimate = tf.reduce_mean(ops.square(rates_rk4 - rates_rk2), axis=1, keepdims=True) + \
        #                  tf.reduce_mean(ops.square(v_avg_rk4 - v_avg_rk2), axis=1, keepdims=True) + \
        #                  tf.reduce_mean(ops.square(w_avg_rk4 - w_avg_rk2), axis=1, keepdims=True)
        #
        # else:
        #     error_estimate = tf.zeros([1], dtype=myconfig.DTYPE)



        return rates_rk4, v_avg_rk4, w_avg_rk4 #, error_estimate


    def call(self, inputs, states):
        rates = states[0]
        v_avg = states[1]
        w_avg = states[2]
        R = states[3]
        U = states[4]
        A = states[5]


        if self.is_nmda:
            gnmda = states[6]
            dgnmda = states[7]

        # integ_error = states[-1]


        g_syn = self.gsyn_max * A
        # g_syn_tot = tf.math.reduce_sum(g_syn, axis=0)
        #
        #
        # Isyn = tf.math.reduce_sum(g_syn * (self.e_r - v_avg), axis=0)

        # if self.is_nmda:
        #     g_syn_nmda = self.gsyn_max_nmda * gnmda / (1 + self.Mgb * exp(-self.av_nmda * (v_avg - 1.0) ) )
        #
        #     Inmda = tf.math.reduce_sum(g_syn_nmda * (self.e_r - v_avg), axis=0)
        #
        #     g_syn_tot += tf.math.reduce_sum(g_syn_nmda, axis=0)
        #
        #     Isyn += Inmda

        # new_rates = rates + self.dts_non_dim * (self.Delta_eta / PI + 2 * rates * v_avg - (self.alpha + g_syn_tot) * rates)
        # new_rates = self.update_rates(v_avg, g_syn_tot, rates)
        # new_rates = tf.where(new_rates < 0, 0.0, new_rates)
        # new_v_avg = v_avg + self.dts_non_dim * (v_avg**2 - self.alpha * v_avg - w_avg + self.I_ext + Isyn - (PI*rates)**2)


        # new_w_avg = w_avg + self.dts_non_dim * (self.a * (self.b * v_avg - w_avg) + self.w_jump * rates)
        # new_w_avg = self.update_w_avg(w_avg, v_avg, rates)

        rates, v_avg, w_avg = self.runge_kutta_step(rates, v_avg, w_avg, g_syn)

        # error_estimate = self.stability_penalty * error_estimate
        #
        # integ_error = 0.2 * integ_error + 0.8 * error_estimate

        firing_probs = tf.transpose( self.dts_non_dim * rates) #tf.reshape(rates, shape=(-1, 1))

        if self.use_input:
            inputs = tf.transpose(inputs) * 0.001 * self.dt_dim
            firing_probs = tf.concat( [firing_probs, inputs], axis=0)

        FRpre_normed = self.pconn *  firing_probs

        tau1r = tf.where(self.tau_d != self.tau_r, self.tau_d / (self.tau_d - self.tau_r), 1e-13)

        exp_tau_d = exp(-self.dt_dim / self.tau_d)
        exp_tau_f = exp(-self.dt_dim / self.tau_f)
        exp_tau_r = exp(-self.dt_dim / self.tau_r)


        a_ = A * exp_tau_d
        r_ = 1 + (R - 1 + tau1r * A) * exp_tau_r  - tau1r * A
        u_ = U * exp_tau_f

        released_mediator = U * r_ * FRpre_normed

        U = u_ + self.Uinc * (1 - u_) * FRpre_normed
        A = a_ + released_mediator
        R = r_ - released_mediator



        if self.is_nmda:
            dgnmda = dgnmda + self.dt_dim * (released_mediator - gnmda - (self.tau1_nmda + self.tau2_nmda )*dgnmda ) / (self.tau1_nmda * self.tau2_nmda)
            gnmda = gnmda + self.dt_dim * dgnmda

            new_states = [rates, v_avg, w_avg, R, U, A, gnmda, dgnmda]

        else:
            new_states = [rates, v_avg, w_avg, R, U, A]

        firings_output = rates * self.dts_non_dim / self.dt_dim * 1000 # convert to spike per second

        output = firings_output

        return output, new_states

    # def update_rates(self, v_avg, g_syn_tot, rates0):
    #     """
    #     """
    #
    #
    #     # Вычисляем показатель экспоненты: shape (N, 1)
    #     exponent_base = -self.alpha - g_syn_tot + 2.0 * v_avg
    #     # exponent_base = tf.expand_dims(exponent_base, axis=1)  # (N, 1)
    #
    #     # Экспонента: exp(t * exponent_base) → shape (N, T)
    #     exponent = exponent_base * self.dts_non_dim  #
    #     exp_term = tf.exp(exponent)  # (N, T)
    #
    #     # Стационарная часть: Delta_eta / (pi * (alpha + g_syn_tot - 2*v_avg))
    #     denominator = self.alpha + g_syn_tot - 2.0 * v_avg
    #     # denominator = tf.expand_dims(denominator, axis=1)  # (N, 1)
    #
    #     # Защита от деления на ноль
    #     safe_denominator = tf.where(
    #         tf.abs(denominator) < 1e-10,
    #         tf.ones_like(denominator),  # временное значение, чтобы не было NaN
    #         denominator
    #     )
    #
    #     stationary = self.Delta_eta / (PI * safe_denominator)  # (N, 1)
    #
    #     # C1 = r0 - stationary; r0 уже (N,), расширяем до (N, 1)
    #     C1 = rates0 - stationary  # (N, 1)
    #
    #     # Итоговое решение: C1 * exp_term + stationary
    #     new_rates = C1 * exp_term + stationary  # (N, T)
    #
    #     return new_rates
    #
    #
    # def update_w_avg(self, w_avg, v_avg, r):
    #     """
    #
    #     """
    #
    #     # Вычисляем стационарную часть: b*v_avg + (r*w_jump)/a
    #     stationary_term_1 = self.b * v_avg  # (N,)
    #
    #     stationary_term_2 = (r * self.w_jump) / self.a
    #
    #     stationary = stationary_term_1 + stationary_term_2  # (N,)
    #
    #     # C1 = w0 - stationary
    #     C1 = w_avg - stationary  # (N, 1)
    #
    #     # Экспоненциальный множитель: exp(-a*t)
    #     exp_term = exp(-self.a * self.dts_non_dim)  # (N, T)
    #
    #     # Итоговое решение
    #     new_w_avg = C1 * exp_term + stationary  # (N, T)
    #
    #     return new_w_avg

    def get_config(self):
        config = super().get_config()
        config.update({
            "dt_dim": self.dt_dim,
            "use_input": self.use_input,

            "gsyn_max": self.gsyn_max,
            "tau_f": self.tau_f,
            "tau_d": self.tau_d,
            "tau_r": self.tau_r,
            "Uinc": self.Uinc,
            "pconn": self.pconn,
            "e_r": self.e_r,

            "alpha": self.alpha,
            "a": self.a,
            "b": self.b,
            "w_jump": self.w_jump,
            "dts_non_dim": self.dts_non_dim,
            "Delta_eta": self.Delta_eta,
            "I_ext": self.I_ext,
        })

        if self.is_nmda:
            nmda_keys = ['pconn_nmda', 'Mgb', 'av_nmda', 'gsyn_max_nmda', 'tau1_nmda', 'tau2_nmda']
            for key in nmda_keys:
                config[key] = getattr(self, key)

        return config

    @classmethod
    def from_config(cls, config):
        ##pprint(config)
        params = {}

        params['gsyn_max'] = config['gsyn_max']['config']["value"]
        params['tau_f'] = config['tau_f']['config']["value"]
        params['tau_d'] = config['tau_d']['config']["value"]
        params['tau_r'] = config['tau_r']['config']["value"]
        params['Uinc'] = config['Uinc']['config']["value"]
        params['pconn'] = config['pconn']['config']["value"]
        params['e_r'] = config['e_r']['config']["value"]

        params['alpha'] = config['alpha']['config']["value"]
        params['a'] = config['a']['config']["value"]
        params['b'] = config['b']['config']["value"]
        params['w_jump'] = config['w_jump']['config']["value"]
        params['dts_non_dim'] = config['dts_non_dim']['config']["value"]
        params['Delta_eta'] = config['Delta_eta']['config']["value"]
        params['I_ext'] = config['I_ext']['config']["value"]

        dt_dim = config['dt_dim']
        use_input = config['use_input']

        if 'gsyn_max_nmda' in config.keys():
            params['nmda'] = {}
            nmda_keys = ['pconn_nmda', 'Mgb', 'av_nmda', 'gsyn_max_nmda', 'tau1_nmda', 'tau2_nmda']

            for key in nmda_keys:
                params['nmda'][key] = config[key]['config']["value"]


        return cls(params, dt_dim=dt_dim, use_input=use_input)

######################################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    NN = 2
    Ninps = 3
    dt_dim = 0.1  # ms
    duration = 1000.0

    dim_izh_params = {
        "V0": -57.63,
        "U0": 0.0,

        "Cm": 114,  # * pF,
        "k": 1.19,  # * mS
        "Vrest": -57.63,  # * mV,
        "Vth": -35.53,  # *mV, # np.random.normal(loc=-35.53, scale=4.0, size=NN) * mV,  # -35.53*mV,
        "Vpeak": 21.72,  # * mV,
        "Vmin": -48.7,  # * mV,
        "a": 0.005,  # * ms ** -1,
        "b": 0.22,  # * mS,
        "d": 2,  # * pA,

        "Iext": 0,  # pA
    }

    # Словарь с константами
    cauchy_dencity_params = {
        'Delta_eta': 15,  # 0.02,
        'bar_eta': 0.0,  # 0.191,
    }

    dim_izh_params = dim_izh_params | cauchy_dencity_params
    izh_params = izhs_lib.dimensional_to_dimensionless(dim_izh_params)
    izh_params['dts_non_dim'] = izhs_lib.transform_T(dt_dim, dim_izh_params['Cm'], dim_izh_params['k'], dim_izh_params['Vrest'])

    for key, val in izh_params.items():
        izh_params[key] = np.zeros(NN, dtype=np.float32) + val

    ## synaptic static variables
    tau_d = 6.02  # ms
    tau_r = 359.8  # ms
    tau_f = 21.0  # ms
    Uinc = 0.25

    gsyn_max = np.zeros(shape=(NN+Ninps, NN), dtype=np.float32)
    # gsyn_max[0, 1] = 20
    # gsyn_max[1, 0] = 15



    pconn = np.zeros(shape=(NN+Ninps, NN), dtype=np.float32)
    # pconn[0, 1] = 1
    # pconn[1, 0] = 1

    Erev = np.zeros(shape=(NN+Ninps, NN), dtype=np.float32) - 75
    e_r = izhs_lib.transform_e_r(Erev, dim_izh_params['Vrest'])

    izh_params['gsyn_max'] = gsyn_max
    izh_params['pconn'] = pconn
    izh_params['e_r'] = np.zeros_like(gsyn_max) + e_r
    izh_params['tau_d'] = np.zeros_like(gsyn_max) + tau_d
    izh_params['tau_r'] = np.zeros_like(gsyn_max) + tau_r
    izh_params['tau_f'] = np.zeros_like(gsyn_max) + tau_f
    izh_params['Uinc'] = np.zeros_like(gsyn_max) + Uinc


    t = tf.range(0, duration, dt_dim, dtype=tf.float32)
    t = tf.reshape(t, shape=(1, -1, 1))

    firings_inputs = tf.zeros(shape=(1, tf.size(t), Ninps), dtype=tf.float32)

    for key, val in izh_params.items():
        print(key, "\n", val)



    meanfieldlayer = MeanFieldNetwork(izh_params, dt_dim=dt_dim, use_input=True)
    meanfieldlayer_rnn = RNN(meanfieldlayer, return_sequences=True, stateful=True)

    input_layer = Input(shape=(None, Ninps), batch_size=1)
    output = meanfieldlayer_rnn(input_layer)

    # output = IntegRegLayer()(output)

    model = Model(inputs=input_layer, outputs=output)

    # model.save(f'test_model.keras')
    # model = load_model('test_model.keras', custom_objects={'MeanFieldNetwork':MeanFieldNetwork, })


    rates = model.predict(firings_inputs)

    print(rates.shape)
    #integ_error = rates[0, :, -1]
    # rates = rates[:, :, :-1].reshape(-1, NN)
    rates = rates.reshape(-1, NN)
    t = t.numpy().ravel()

    plt.plot(t, rates)
    plt.show()

    # plt.plot(t, integ_error)
    # plt.show()