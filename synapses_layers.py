import tensorflow as tf
from tensorflow.keras.layers import Layer
exp = tf.math.exp

# Задачи
# * Задать начальное состояние
# * Сделать предвычисление коэфициетов exp(-dt/tau)

class BaseSynapse(Layer):
    ## В классе реализовать запоминание
    # *Erev для каждого синапса,
    # *Маски для выбора нужных входов
    # *сжатия входов до двух
    # * Реализовать метод call


    def __init__(self, params, dt=0.1, mask=None, **kwargs):
        super(BaseSynapse, self).__init__(**kwargs)
        self.dt = tf.convert_to_tensor(dt)
        self.pconn = tf.convert_to_tensor( params['pconn'] )
        self.Erev = tf.convert_to_tensor( params['Erev'] )
        self.Cm = tf.convert_to_tensor( params['Cm'] )

        if mask is None:
            self.mask = tf.ones([self.units, ], dtype=tf.dtypes.bool)
        else:
            self.mask = tf.convert_to_tensor(mask)

        self.units = tf.size(self.pconn)

        self.output_size = 2
        self.state_size = []

    def build(self, input_shape):
        super().build(input_shape)

        self.built = True

    # def call(self, inputs, states):
    #
    #     return s, s


class TsodycsMarkramSynapse(Layer):

    def __init__(self, params, dt=0.1, **kwargs):
        super(TsodycsMarkramSynapse, self).__init__(params, dt=dt, **kwargs)

        self.gsyn_max = tf.Variable( params['gsyn_max'], name="gsyn_max", trainable=True )
        self.tau_f = tf.Variable( params['tau_f'], name="tau_f", trainable=False)
        self.tau_d = tf.Variable( params['tau_d'], name="tau_d", trainable=False )
        self.tau_r = tf.Variable( params['tau_r'], name="tau_r", trainable=False )
        self.Uinc  = tf.Variable( params['Uinc'], name="Uinc", trainable=False )

        self.tau1r = tf.where(self.tau_d != self.tau_r, self.tau_d / (self.tau_d - self.tau_r), 1e-13)
        self.state_size = [self.units, self.units, self.units]


    def build(self, input_shape):
        super().build(input_shape)
        self.built = True


    def call(self, inputs, states):
        FR = tf.boolean_mask(inputs, self.mask)
        R = states[0]
        U = states[1]
        X = states[2]

        FRpre_normed =  FR * self.pconn

        y_ = R * exp(-self.dt / self.tau_d)

        x_ = 1 + (X - 1 + self.tau1r * U) * exp(-self.dt / self.tau_r) - self.tau1r * U

        u_ = U * exp(-self.dt / self.tau_f)
        U = u_ + self.Uinc * (1 - u_) * FRpre_normed
        R = y_ + U * x_ * FRpre_normed
        X = x_ - U * x_ * FRpre_normed

        gsyn = self.gsyn_max * X

        g_tot = tf.reduce_sum(gsyn)
        E = tf.reduce_sum(gsyn * self.Erev) / g_tot
        tau = self.Cm / g_tot

        output = tf.concat([E, tau])


        return output, [R, U, X]