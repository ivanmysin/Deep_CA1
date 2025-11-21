import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


os.chdir('../')

from genloss import PhaseLockingOutputWithPhase, PhaseLockingOutput
from np_meanfield import SpatialThetaGenerators

params = [
        {
                "R": 0.25,
                "OutPlaceFiringRate": 0.5,
                "OutPlaceThetaPhase": np.random.uniform(-np.pi, np.pi),
                "InPlacePeakRate": 8.0,
                "CenterPlaceField": -5000.0,
                "SigmaPlaceField": 500,
                "SlopePhasePrecession": 0.0, # np.deg2rad(10)*10 * 0.001,
                "PrecessionOnset":  -1.57,
                "ThetaFreq": 8.0,
        },

        {
                "R": 0.3,
                "OutPlaceFiringRate": 20.0,
                "OutPlaceThetaPhase": np.random.uniform(-np.pi, np.pi),
                "InPlacePeakRate": 8.0,
                "CenterPlaceField": -5000.0,
                "SigmaPlaceField": 500,
                "SlopePhasePrecession": 0.0, # np.deg2rad(10)*10 * 0.001,
                "PrecessionOnset":  -1.57,
                "ThetaFreq": 8.0,
        },


        {
                "R": 0.45,
                "OutPlaceFiringRate": 5.0,
                "OutPlaceThetaPhase": np.random.uniform(-np.pi, np.pi),
                "InPlacePeakRate": 8.0,
                "CenterPlaceField": -5000.0,
                "SigmaPlaceField": 500,
                "SlopePhasePrecession": 0.0, # np.deg2rad(10)*10 * 0.001,
                "PrecessionOnset":  -1.57,
                "ThetaFreq": 8.0,
        },
]

generators = SpatialThetaGenerators(params)


dt = 0.1
t = np.arange(0, 120, dt).reshape(-1, 1)

firings = generators.call(t.reshape(1, -1, 1))
#
# phi0 = np.asarray( [0.0, 0.5*np.pi, np.pi]).reshape(1, -1)
# firings = 0.5*(np.cos(2*np.pi*8*t*0.001 + phi0) + 1.0)  # 0.5 + 0.5 *
# MeanFirings = np.asarray( [0.5, 1.5, 6.0])


MeanFirings = [p['OutPlaceFiringRate'] for p in params]
MeanFirings = np.asarray(MeanFirings).reshape(1, 1, -1)

Rs = [p['R'] for p in params]
Rs = np.asarray(Rs).reshape(1, -1)

Phases = [p['OutPlaceThetaPhase'] for p in params]
Phases = np.asarray(Phases).reshape(1, -1)

furie_trst_targets = np.stack( [Rs * np.cos(Phases), Rs * np.sin(Phases)], axis=1)



# phase_locking_layer = PhaseLockingOutputWithPhase( MeanFirings, ThetaFreq=8.0, dt=dt)


input = Input(shape=(None, 3), batch_size=1)

phase_locking_layer = PhaseLockingOutputWithPhase( MeanFirings, ThetaFreq=8.0, dt=dt)(input)
model = Model(inputs=input, outputs=phase_locking_layer)
model.compile(
        optimizer='adam',
        loss = 'mse',
)

furie_trst = model.predict(firings)


print(furie_trst.shape)
print(furie_trst_targets.shape)

# furie_abs = np.sqrt(furie_trst[0, 0, :]**2 +  furie_trst[0, 1, :]**2)

print(furie_trst_targets)
print('=====================')
print(furie_trst)

print('=====================')

L = np.mean( np.log(  (furie_trst_targets+1) / (furie_trst+1) )**2 )

print(L)
#plt.plot(t, firings[0])
#plt.show()
