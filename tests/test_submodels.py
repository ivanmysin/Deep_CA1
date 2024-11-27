import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.src.backend import shape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, RNN, Layer, Reshape
from tensorflow.keras.saving import load_model

from pprint import pprint
import os

import myconfig

os.chdir("../")
from synapses_layers import TsodycsMarkramSynapse
from genloss import SpatialThetaGenerators, CommonOutProcessing, PhaseLockingOutputWithPhase, PhaseLockingOutput, RobastMeanOut, RobastMeanOutRanger, Decorrelator


class TimeStepLayer(Layer):

    def __init__(self, units, base_pop_model, synapses_params, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

        self.base_pop_model = base_pop_model
        self.synapses_params = synapses_params


    def get_initial_state(self, batch_size=1):
        state = K.zeros(shape=(batch_size, self.state_size), dtype=tf.float32)
        return state

    def get_pop_model_with_synapses(self, input_shape, syn_params):

        dt = 0.1

        ndimsinp = syn_params["gsyn_max"].size

        mask = np.ones(ndimsinp, dtype=bool)

        synapses = TsodycsMarkramSynapse(syn_params, dt=dt, mask=mask)
        synapses_layer = RNN(synapses, return_sequences=True, stateful=True, name="Synapses_Layer")

        input_layer = Input(shape=(None, ndimsinp), batch_size=1)
        synapses_layer = synapses_layer(input_layer)

        base_model = tf.keras.models.clone_model(self.base_pop_model)

        model = Model(inputs=input_layer, outputs=base_model(synapses_layer), name="Population_with_synapses")

        return model



    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.n_dims = input_shape[-1]

        self.pop_models = []
        for syn_params in self.synapses_params:
            model = self.get_pop_model_with_synapses(input_shape, syn_params)
            self.pop_models.append(model)

    def call(self, input, state):


        input = K.concatenate([state[0], input], axis=-1)
        input = K.reshape(input, shape=(1, 1, -1))

        output = []
        for model in self.pop_models:
            out = model(input)
            output.append(out)
        output = K.concatenate(output, axis=-1)

        return output, output[0]


# # Параметры данных
Ns = 5
ext_input = 3

full_size = Ns+ext_input

params = {
    "gsyn_max": np.zeros(full_size, dtype=np.float32) + 1.5,
    "Uinc": np.zeros(full_size, dtype=np.float32) + 0.5,
    "tau_r": np.zeros(full_size, dtype=np.float32) + 1.5,
    "tau_f": np.zeros(full_size, dtype=np.float32) + 1.5,
    "tau_d": np.zeros(full_size, dtype=np.float32) + 1.5,
    'pconn': np.zeros(full_size, dtype=np.float32) + 1.0,
    'Erev': np.zeros(full_size, dtype=np.float32),
    'Cm': 0.114,
    'Erev_min': -75.0,
    'Erev_max': 0.0,
}

spatial_gen_params = [
    {
        "OutPlaceFiringRate" : 0.5,
        "OutPlaceThetaPhase": 3.14,
        "InPlacePeakRate" : 8.0,
        "R" : 0.3,
        "CenterPlaceField" : 200,
        "SigmaPlaceField" : 50,
        "SlopePhasePrecession" : 10,
        "PrecessionOnset" : -1.5,
        "ThetaFreq" : 6.0,
    },
    {
        "OutPlaceFiringRate": 0.5,
        "OutPlaceThetaPhase": 3.14,
        "InPlacePeakRate": 8.0,
        "R": 0.3,
        "CenterPlaceField": 200,
        "SigmaPlaceField": 50,
        "SlopePhasePrecession": 10,
        "PrecessionOnset": -1.5,
        "ThetaFreq": 6.0,
    },
    {
        "OutPlaceFiringRate": 0.5,
        "OutPlaceThetaPhase" : 3.14,
        "InPlacePeakRate": 8.0,
        "R": 0.3,
        "CenterPlaceField": 200,
        "SigmaPlaceField": 50,
        "SlopePhasePrecession": 10,
        "PrecessionOnset": -1.5,
        "ThetaFreq": 6.0,
    },

]

ints_phases = [{"ThetaPhase": 3.14} for _ in range(Ns)]


synapse_params = [params for _ in range(Ns)]

base_model = load_model("./pretrained_models/NO_Trained.keras")
for layer in base_model.layers:
    layer.trainable = False

time_step_layer = TimeStepLayer(Ns, base_model, synapse_params)
time_step_layer = RNN(time_step_layer, return_sequences=True, stateful=True, activity_regularizer=RobastMeanOutRanger() )

input = Input(shape=(None, 1), batch_size=1)
generators = SpatialThetaGenerators(spatial_gen_params)(input)

time_step_layer = time_step_layer(generators)

time_step_layer = Reshape(target_shape=(-1, Ns), activity_regularizer=Decorrelator(strength=0.1))(time_step_layer)



output_layers = []
simple_out_mask = np.ones(Ns, dtype='bool')
simple_selector = CommonOutProcessing(simple_out_mask, name='pyramilad_mask')
output_layers.append(simple_selector(time_step_layer) )

frequecy_filter_out_mask = np.ones(Ns, dtype='bool')
theta_phase_locking_with_phase = PhaseLockingOutputWithPhase(mask=frequecy_filter_out_mask,\
                                                                     ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT, name='locking_with_phase')
output_layers.append(theta_phase_locking_with_phase(time_step_layer))


robast_mean_out = RobastMeanOut(mask=frequecy_filter_out_mask, name='robast_mean')
output_layers.append(robast_mean_out(time_step_layer))

phase_locking_out_mask = np.ones(Ns, dtype='bool')
phase_locking_selector = PhaseLockingOutput(mask=phase_locking_out_mask,
                                                    ThetaFreq=myconfig.ThetaFreq, dt=myconfig.DT, name='locking')
output_layers.append(phase_locking_selector(time_step_layer))

big_model = Model(inputs=input, outputs=output_layers)
# big_model.build(input_shape = (None, 1))

big_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss={
        'pyramilad_mask' : tf.keras.losses.logcosh,
        'locking_with_phase' : tf.keras.losses.MSE,
        'robast_mean' : tf.keras.losses.MSE,
        'locking' : tf.keras.losses.MSE,
    }
)

print(big_model.summary())


timesteps = 100
# Генерация случайных входных данных
# X = np.random.rand(1, timesteps, ext_input)
dt = myconfig.DT
X = np.arange(0, timesteps*dt, dt).reshape(1, -1, 1)

# Генерация ответов
Ys = {
    'pyramilad_mask' : np.random.rand(timesteps, Ns).reshape(1, timesteps, Ns),
    'locking_with_phase': np.random.rand(2, Ns).reshape(1, 2, Ns),
    'robast_mean' : np.random.rand(Ns).reshape(1, 1, Ns),
    'locking' : np.random.rand(Ns).reshape(1, 1, Ns),
}


X = tf.convert_to_tensor(value=X, dtype='float32')
y_preds = big_model.predict(X)
for y_pred in y_preds:
    print(y_pred.shape)



hist = big_model.fit(X, Ys, epochs=2, batch_size=1, verbose=2)



