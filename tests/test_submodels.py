import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, RNN, Layer

class TimeStepLayer(Layer):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units


    # def get_initial_state(self, batch_size=None):
    #     return K.zeros([batch_size, self.state_size])

    def build(self, input_shape):

        self.batch_size = input_shape[0]
        self.n_dims = input_shape[-1]

        self.pop_models = []
        for _ in range(self.units):
            model = Sequential()
            model.add(Input(shape=(None, input_shape[-1]+self.state_size)))
            model.add(GRU(16, return_sequences=True, stateful=False) )  #
            model.add(GRU(16, return_sequences=True, stateful=False))  #
            model.add(Dense(1, activation='relu'))  #
            self.pop_models.append(model)

    def call(self, input, state):

        input = K.reshape(input, shape=(1, 1, self.n_dims))
        input = K.concatenate([input, state], axis=2)

        output = []
        for model in self.pop_models:
            out = model(input)
            output.append(out)
        output = K.concatenate(output, axis=2)

        return output, output[0]

input_shape = (None, 7)
model = Sequential()
model.add(Input(shape=(None, input_shape[-1])))
model.add(GRU(16, return_sequences=True, stateful=False) )  #
model.add(GRU(16, return_sequences=True, stateful=False))  #
model.add(Dense(1, activation='relu'))


X = np.random.rand(1, 10, input_shape[-1])
Ypred = model.predict(X)
print(Ypred.shape)


# # # Параметры данных
# Ns = 5
# input_shape = (1, None, 2)
# timesteps = 100
# # Генерация случайных входных данных
# X = np.random.rand(input_shape[0], timesteps, input_shape[-1])
# # Генерация меток (например, бинарная классификация)
# Y = np.random.rand(timesteps, 5).reshape(1, 100, 1, 5)
#
#
# big_model = Sequential()
# my_layer = RNN(TimeStepLayer(Ns), return_sequences=True, stateful=False)
# big_model.add(my_layer)
# big_model.build(input_shape=input_shape)
# big_model.compile(loss="mean_squared_logarithmic_error", optimizer="adam")
#
# print(big_model.trainable_variables)
#
# hist = big_model.fit(X, Y, epochs=2)
#
# #print(y_pred.shape)



