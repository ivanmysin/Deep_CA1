import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, RNN, Layer, Reshape
from tensorflow.keras.saving import load_model


old_base_model = load_model('../pretrained_models/CA1 Basket.keras', custom_objects={'square': tf.keras.ops.square})

model = Sequential()
model.add(Input(shape=(None, 1), batch_size=1))

for layer in old_base_model.layers:
    layer_type = layer.__class__.__name__

    print(layer_type)
    print(layer.units)

    Layer_obj = getattr(tf.keras.layers, layer_type)

    if (layer_type == 'GRU') or (layer_type == 'LSTM'):
        model.add(Layer_obj(units = layer.units, return_sequences=True, stateful=True)  )
    else:
        model.add(Layer_obj(units=layer.units, activation=layer.activation))

model.build()

for newlayer, oldlayer in zip(model.layers, old_base_model.layers):
    newlayer.set_weights(oldlayer.get_weights())
    newlayer.trainable = False


# model.add(tf.keras.models.clone_model(population_model.layers[0]))
# model.add(tf.keras.models.clone_model(population_model.layers[1]))
# # model.layers[0].trainable = True
# model.layers[1].trainable = False
# model.layers[2].trainable = False
#
# model.build(input_shape=input_shape)

