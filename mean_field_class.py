import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN, Input
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.saving import load_model
import izhs_lib
import h5py

import sys
sys.path.append('../')
import myconfig

#

PI = 3.141592653589793
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

    def __init__(self, lw=0.01, close_coeff=100):
        self.close_coeff = close_coeff
        self.lw = lw

    def __call__(self, x):
        return tf.reduce_sum(-self.lw * tf.math.log(x * self.close_coeff))

    def get_config(self):
        config = {
            "close_coeff" : float(self.close_coeff),
            "lw" : float(self.lw),
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

    def __init__(self, min=0, max=10000):
        self.min = min
        self.max = max


    def __call__(self, w):
        return tf.clip_by_value(w, clip_value_min=self.min, clip_value_max=self.max)

    def get_config(self):
        config = {
            "min" : float(self.min),
            "max" : float(self.max),
        }

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.keras.utils.register_keras_serializable(package="MeanFieldNetwork")
class MeanFieldNetwork(Layer):

    def __init__(self, params, dt_dim=0.5, use_input=False, **kwargs):
        super().__init__(**kwargs)

        self.dt_dim = dt_dim
        self.use_input = use_input

        self.units = len(params['alpha'])
        self.alpha = tf.convert_to_tensor( params['alpha'], dtype=myconfig.DTYPE )
        self.a = tf.convert_to_tensor( params['a'], dtype=myconfig.DTYPE )
        self.b = tf.convert_to_tensor( params['b'], dtype=myconfig.DTYPE )
        self.w_jump = tf.convert_to_tensor( params['w_jump'], dtype=myconfig.DTYPE )
        self.dts_non_dim = tf.convert_to_tensor( params['dts_non_dim'], dtype=myconfig.DTYPE )
        self.Delta_eta = tf.convert_to_tensor( params['Delta_eta'], dtype=myconfig.DTYPE)

        I_ext = tf.convert_to_tensor( params['I_ext'], dtype=myconfig.DTYPE )
        self.I_ext = self.add_weight(shape=tf.keras.ops.shape(I_ext),
                                        initializer=tf.keras.initializers.Constant(I_ext),
                                        trainable=True,
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
                                        constraint=MinMaxWeights(min=0, max=1),
                                        name=f"pconn")


        self.gsyn_max = self.add_weight(shape=tf.keras.ops.shape(gsyn_max),
                                        initializer=tf.keras.initializers.Constant(gsyn_max),
                                        # regularizer=self.gmax_regulizer,  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        trainable=True,
                                        dtype=myconfig.DTYPE,
                                        constraint=tf.keras.constraints.NonNeg(),
                                        name=f"gsyn_max")

        self.tau_f = self.add_weight(shape=tf.keras.ops.shape(tau_f),
                                     initializer=tf.keras.initializers.Constant(tau_f),
                                     # regularizer=ZeroWallReg(lw=0.001, close_coeff=1000),
                                     trainable=True,
                                     dtype=myconfig.DTYPE,
                                     constraint=MinMaxWeights(min=dt_dim), # tf.keras.constraints.NonNeg(),
                                     name=f"tau_f")

        self.tau_d = self.add_weight(shape=tf.keras.ops.shape(tau_d),
                                     initializer=tf.keras.initializers.Constant(tau_d),
                                     # regularizer=ZeroWallReg(lw=0.001, close_coeff=1000),
                                     trainable=True,
                                     dtype=myconfig.DTYPE,
                                     constraint=MinMaxWeights(min=dt_dim), # tf.keras.constraints.NonNeg(),
                                     name=f"tau_d")

        self.tau_r = self.add_weight(shape=tf.keras.ops.shape(tau_r),
                                     initializer=tf.keras.initializers.Constant(tau_r),
                                     # regularizer=ZeroWallReg(lw=0.001, close_coeff=1000),
                                     trainable=True,
                                     dtype=myconfig.DTYPE,
                                     constraint=MinMaxWeights(min=dt_dim),  #tf.keras.constraints.NonNeg(),
                                     name=f"tau_r")

        self.Uinc = self.add_weight(shape=tf.keras.ops.shape(Uinc),
                                    initializer=tf.keras.initializers.Constant(Uinc),
                                    # regularizer=ZeroOneWallReg(lw=0.001, close_coeff=1000),
                                    trainable=True,
                                    dtype=myconfig.DTYPE,
                                    constraint=MinMaxWeights(min=0, max=1),  # ZeroOnesWeights(),
                                    name=f"Uinc")

        synaptic_matrix_shapes = tf.shape(self.gsyn_max)

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

        R = tf.ones( synaptic_matrix_shapes, dtype=myconfig.DTYPE)
        U = tf.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)
        A = tf.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)

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
        v_avg = v_avg + self.dts_non_dim * (v_avg**2 - self.alpha * v_avg - w_avg + self.I_ext + Isyn - (PI*rates)**2)
        w_avg = w_avg + self.dts_non_dim * (self.a * (self.b * v_avg - w_avg) + self.w_jump * rates)

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

        U = u_ + self.Uinc * (1 - u_) * FRpre_normed
        A = a_ + U * r_ * FRpre_normed
        R = r_ - U * r_ * FRpre_normed


        output = rates * self.dts_non_dim / self.dt_dim * 1000 # convert to spike per second
        #output = v_avg

        return output, [rates, v_avg, w_avg, R, U, A]

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
        'Delta_eta': 80,  # 0.02,
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

    model = Model(inputs=input_layer, outputs=output)

    # model.save(f'test_model.keras')
    # model = load_model('test_model.keras', custom_objects={'MeanFieldNetwork':MeanFieldNetwork, })


    rates = model.predict(firings_inputs)

    rates = rates.reshape(-1, NN)
    t = t.numpy().ravel()

    plt.plot(t, rates)
    plt.show()