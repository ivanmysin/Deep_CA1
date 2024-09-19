import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN
exp = tf.math.exp

# Задачи
# * Задать начальное состояние


class BaseSynapse(Layer):

    def __init__(self, params, dt=0.1, mask=None, **kwargs):
        super(BaseSynapse, self).__init__(**kwargs)
        self.dt = tf.convert_to_tensor(dt, dtype=tf.float32)
        self.pconn = tf.convert_to_tensor( params['pconn'], dtype=tf.float32 )
        self.Erev = tf.convert_to_tensor( params['Erev'], dtype=tf.float32 )
        self.Cm = tf.convert_to_tensor( params['Cm'], dtype=tf.float32 )
        self.Erev_min = tf.convert_to_tensor( params['Erev_min'], dtype=tf.float32 )
        self.Erev_max = tf.convert_to_tensor( params['Erev_max'], dtype=tf.float32 )

        self.units = tf.size(self.pconn)

        if mask is None:
            self.mask = tf.ones([self.units, ], dtype=tf.dtypes.bool)
        else:
            self.mask = tf.convert_to_tensor(mask)



        self.output_size = 2
        self.state_size = []

    def build(self, input_shape):
        super().build(input_shape)

        self.built = True



class TsodycsMarkramSynapse(BaseSynapse):

    def __init__(self, params, dt=0.1, mask=None, **kwargs):

        super(TsodycsMarkramSynapse, self).__init__(params, dt=dt, mask=mask, **kwargs)

        self.gsyn_max = tf.keras.Variable( params['gsyn_max'], name="gsyn_max", trainable=True, dtype=tf.float32 )
        self.tau_f = tf.keras.Variable( params['tau_f'], name="tau_f", trainable=False, dtype=tf.float32)
        self.tau_d = tf.keras.Variable( params['tau_d'], name="tau_d", trainable=False, dtype=tf.float32 )
        self.tau_r = tf.keras.Variable( params['tau_r'], name="tau_r", trainable=False, dtype=tf.float32 )
        self.Uinc  = tf.keras.Variable( params['Uinc'], name="Uinc", trainable=False, dtype=tf.float32 )

        self.tau1r = tf.where(self.tau_d != self.tau_r, self.tau_d / (self.tau_d - self.tau_r), 1e-13)
        self.state_size = [self.units, self.units, self.units]

        self.exp_tau_d = exp(-self.dt / self.tau_d)
        self.exp_tau_f = exp(-self.dt / self.tau_f)
        self.exp_tau_r = exp(-self.dt / self.tau_r)

    # def build(self, input_shape):
    #     super(TsodycsMarkramSynapse, self).build(input_shape)
    #
    #     #self.gsyn_max = tf.Variable(self.gsyn_max, name="gsyn_max", trainable=True, dtype=tf.float32)
    #
    #     self.built = True


    def call(self, inputs, states):
        FR = tf.boolean_mask(inputs, self.mask, axis=1)
        R = states[0]
        U = states[1]
        A = states[2]

        FRpre_normed =  FR * self.pconn

        a_ = A * self.exp_tau_d
        r_ = 1 + (R - 1 + self.tau1r * A) * self.exp_tau_r  - self.tau1r * A
        u_ = U * self.exp_tau_f

        U = u_ + self.Uinc * (1 - u_) * FRpre_normed
        A = a_ + U * r_ * FRpre_normed
        R = r_ - U * r_ * FRpre_normed


        gsyn = tf.nn.relu(self.gsyn_max) * A

        g_tot = tf.reduce_sum(gsyn, axis=-1)
        E = tf.reduce_sum(gsyn * self.Erev, axis=-1) / g_tot

        E = (E - self.Erev_min) / (self.Erev_max - self.Erev_min) #- 1

        tau = self.Cm / g_tot

        tau = tf.math.log(tau + 1.0)

        output = tf.stack([E, tau], axis=-1)

        return output, [R, U, A]

    def get_initial_state(self, batch_size=None):
        shape = [batch_size, self.units]

        #print(batch_size)

        R = tf.ones( shape, dtype=tf.float32)
        U = tf.zeros( shape, dtype=tf.float32)
        A = tf.zeros( shape, dtype=tf.float32)
        initial_state = [R, U, A]

        return initial_state

if __name__ == "__main__":
    import numpy as np

    Ns = 5
    params = {
        "gsyn_max" : np.zeros(Ns, dtype=np.float32) + 1.5,
        "Uinc" :  np.zeros(Ns, dtype=np.float32) + 0.5,
        "tau_r" : np.zeros(Ns, dtype=np.float32) + 1.5,
        "tau_f" : np.zeros(Ns, dtype=np.float32) + 1.5,
        "tau_d" : np.zeros(Ns, dtype=np.float32) + 1.5,
        'pconn' : np.zeros(Ns, dtype=np.float32) + 1.0,
        'Erev' : np.zeros(Ns, dtype=np.float32),
        'Cm' : 0.114,
    }
    dt = 0.1
    mask = np.ones(Ns, dtype=bool)

    input_shape = (1, None, Ns)

    synapses = TsodycsMarkramSynapse(params, dt=dt, mask=mask)

    synapses_layer = RNN(synapses, return_sequences=True, stateful=True)

    model = tf.keras.Sequential()
    model.add(synapses_layer)

    model.build(input_shape=input_shape)



    X = np.random.rand(50).reshape(1, 10, 5)

    Y = model.predict(X)

    print(Y.shape)