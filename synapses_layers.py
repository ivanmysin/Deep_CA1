import tensorflow as tf
import myconfig
from tensorflow.keras.layers import Layer, RNN

exp = tf.math.exp
tf.keras.backend.set_floatx(myconfig.DTYPE)
from pprint import pprint


class BaseSynapse(Layer):

    def __init__(self, params, dt=0.1, mask=None, **kwargs):
        super(BaseSynapse, self).__init__(**kwargs)
        self.dt = tf.convert_to_tensor(dt, dtype=myconfig.DTYPE)
        self.pconn = tf.convert_to_tensor( params['pconn'], dtype=myconfig.DTYPE )
        self.Erev = tf.convert_to_tensor( params['Erev'], dtype=myconfig.DTYPE )
        self.Cm = tf.convert_to_tensor( params['Cm'], dtype=myconfig.DTYPE )
        self.Erev_min = tf.convert_to_tensor( params['Erev_min'], dtype=myconfig.DTYPE )
        self.Erev_max = tf.convert_to_tensor( params['Erev_max'], dtype=myconfig.DTYPE )
        self.Vrest = tf.convert_to_tensor( params['Vrest'], dtype=myconfig.DTYPE )
        self.gl = tf.convert_to_tensor( params['gl'], dtype=myconfig.DTYPE )

        try:
            self.pop_idx = params['pop_idx']
        except:
            self.pop_idx = 0


        self.units = tf.size(self.pconn)

        if mask is None:
            self.mask = tf.ones([self.units, ], dtype=tf.dtypes.bool)
        else:
            self.mask = tf.convert_to_tensor(mask)



        self.output_size = 1
        self.state_size = []

    def build(self, input_shape):
        super().build(input_shape)

        self.built = True



class TsodycsMarkramSynapse(BaseSynapse):

    def __init__(self, params, dt=0.1, mask=None, **kwargs):

        super(TsodycsMarkramSynapse, self).__init__(params, dt=dt, mask=mask, **kwargs)

        #self.gsyn_max = tf.keras.Variable( params['gsyn_max'], name="gsyn_max", trainable=True, dtype=myconfig.DTYPE )
        self.gmax_regulizer = tf.keras.regularizers.L2(l2=0.001)
        gsyn_max = tf.convert_to_tensor(params['gsyn_max'])

        self.gsyn_max = self.add_weight(shape = tf.keras.ops.shape(gsyn_max),
                                        initializer = tf.keras.initializers.Constant(gsyn_max),
                                        regularizer = self.gmax_regulizer,
                                        trainable = True,
                                        dtype = myconfig.DTYPE,
                                        constraint = tf.keras.constraints.NonNeg(),
                                        name = f"gsyn_max_{self.pop_idx}")


        #self.gsyn_max = tf.keras.Variable( params['gsyn_max'], name="gsyn_max", trainable=True, dtype=myconfig.DTYPE)
        self.tau_f = tf.keras.Variable( params['tau_f'], name="tau_f", trainable=False, dtype=myconfig.DTYPE)
        self.tau_d = tf.keras.Variable( params['tau_d'], name="tau_d", trainable=False, dtype=myconfig.DTYPE )
        self.tau_r = tf.keras.Variable( params['tau_r'], name="tau_r", trainable=False, dtype=myconfig.DTYPE )
        self.Uinc  = tf.keras.Variable( params['Uinc'], name="Uinc", trainable=False, dtype=myconfig.DTYPE )

        self.state_size = [self.units, self.units, self.units, 1]

        #assert(self.mask.numpy().sum() > 0)
        # print("dt =", self.dt.numpy())
        # pprint(params)



    def build(self, input_shape):
        super(TsodycsMarkramSynapse, self).build(input_shape)

        self.tau1r = tf.where(self.tau_d != self.tau_r, self.tau_d / (self.tau_d - self.tau_r), 1e-13)
        #self.state_size = [self.units, self.units, self.units, 1]

        self.exp_tau_d = exp(-self.dt / self.tau_d)
        self.exp_tau_f = exp(-self.dt / self.tau_f)
        self.exp_tau_r = exp(-self.dt / self.tau_r)

        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "gsyn_max": self.gsyn_max,
            "tau_f": self.tau_f,
            "tau_d": self.tau_d,
            "tau_r": self.tau_r,
            "Uinc": self.Uinc,
            "pconn": self.pconn,
            "Erev": self.Erev,
            "Erev_min": self.Erev_min,
            "Erev_max": self.Erev_max,
            "Vrest": self.Vrest,
            "gl": self.gl,
            "Cm": self.Cm,
            "dt" : self.dt,
            "mask" : self.mask,
            "pop_idx" : self.pop_idx
        })
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
        params['Erev'] = config['Erev']['config']["value"]
        params['Erev_min'] = config['Erev_min']['config']["value"]
        params['Erev_max'] = config['Erev_max']['config']["value"]
        params['Vrest'] = config['Vrest']['config']["value"]
        params['gl'] = config['gl']['config']["value"]
        params['Cm'] = config['Cm']['config']["value"]
        params['pop_idx'] = config['pop_idx']
        dt = config['dt']['config']["value"]
        mask = config['mask']['config']["value"]

        return cls(params, dt=dt, mask=mask)


    def call(self, inputs, states):



        FR = tf.boolean_mask(inputs, self.mask, axis=1)

        R = states[0]
        U = states[1]
        A = states[2]
        Vsyn = states[3]

        FRpre_normed =  FR * self.pconn * 0.001 * self.dt # to convert firings in Hz to probability

        a_ = A * self.exp_tau_d
        r_ = 1 + (R - 1 + self.tau1r * A) * self.exp_tau_r  - self.tau1r * A
        u_ = U * self.exp_tau_f

        U = u_ + self.Uinc * (1 - u_) * FRpre_normed
        A = a_ + U * r_ * FRpre_normed
        R = r_ - U * r_ * FRpre_normed


        gsyn = tf.nn.relu(self.gsyn_max) * A
        g_tot = tf.reduce_sum(gsyn, axis=-1) + self.gl
        gE = gsyn * self.Erev
        E_inf = (tf.reduce_sum(gE, axis=-1) + self.gl*self.Vrest ) / g_tot
        tau = self.Cm / g_tot

        Vsyn =  Vsyn - (Vsyn - E_inf) * (1 - exp(-self.dt / tau))
        output = (Vsyn - self.Erev_min) / (self.Erev_max - self.Erev_min)
        output = tf.reshape(output, shape=(1, 1))

        #print(output.Vsyn())

        return output, [R, U, A, Vsyn]

    def get_initial_state(self, batch_size=1):
        shape = [batch_size, self.units]

        R = tf.ones( shape, dtype=myconfig.DTYPE)
        U = tf.zeros( shape, dtype=myconfig.DTYPE)
        A = tf.zeros( shape, dtype=myconfig.DTYPE)

        Vsyn = tf.zeros((batch_size, 1), dtype=myconfig.DTYPE) + self.Vrest



        initial_state = [R, U, A, Vsyn]

        return initial_state

    # def add_regularization_penalties(self):
    #     self.add_loss(self.gmax_regulizer(self.gsyn_max))


if __name__ == "__main__":
    import numpy as np

    Ns = 5
    params = {
        "gsyn_max" : np.zeros(Ns, dtype=myconfig.DTYPE) + 1.5,
        "Uinc" :  np.zeros(Ns, dtype=myconfig.DTYPE) + 0.5,
        "tau_r" : np.zeros(Ns, dtype=myconfig.DTYPE) + 1.5,
        "tau_f" : np.zeros(Ns, dtype=myconfig.DTYPE) + 1.5,
        "tau_d" : np.zeros(Ns, dtype=myconfig.DTYPE) + 1.5,
        'pconn' : np.zeros(Ns, dtype=myconfig.DTYPE) + 1.0,
        'Erev' : np.zeros(Ns, dtype=myconfig.DTYPE),
        'Vrest' : np.zeros(1, dtype=myconfig.DTYPE) - 65.0,
        'Erev_min' : -75.0,
        'Erev_max' : 0.0,
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