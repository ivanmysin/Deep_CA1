import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


os.chdir('../')

from genloss import PhaseLockingOutputWithPhase, PhaseLockingOutput

dt = 0.1
t = np.arange(0, 240, dt).reshape(-1, 1)
firings = 0.5 + 0.5 * np.cos(2*np.pi*8*t*0.001)


MeanFirings = np.asarray( [0.5, 1.5, 8.0])

MeanFirings = MeanFirings.reshape(1, -1)

firings = firings * MeanFirings

firings = firings.reshape(1, firings.shape[0], firings.shape[1])

print(firings.shape)


# phase_locking_layer = PhaseLockingOutputWithPhase( MeanFirings, ThetaFreq=8.0, dt=dt)


input = Input(shape=(None, 3), batch_size=1)

phase_locking_layer = PhaseLockingOutputWithPhase( MeanFirings, ThetaFreq=8.0, dt=dt)(input)
model = Model(inputs=input, outputs=phase_locking_layer)
model.compile(
        optimizer='adam',
        loss = 'mse',
)

furie_trst = model.predict(firings)


print(furie_trst)
#plt.plot(t, firings[0])
#plt.show()
