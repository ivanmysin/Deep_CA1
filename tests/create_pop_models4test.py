from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.saving import load_model


model = Sequential()
model.add(Input(shape=(None, 2)))
model.add(GRU(16, return_sequences=True, kernel_initializer=keras.initializers.HeUniform()))  # , stateful=True
model.add(GRU(16, return_sequences=True, kernel_initializer=keras.initializers.HeUniform()))  # , stateful=True
model.add(Dense(1, activation='relu'))  #

model.compile(loss="mean_squared_logarithmic_error", optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['mean_squared_logarithmic_error'])


model.save("../pretrained_models/NO_Trained.keras")