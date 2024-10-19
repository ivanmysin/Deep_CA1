import numpy as np
import sys
sys.path.append("../")
import genloss


firings = np.random.rand(1000).reshape(1, 100, 10)
t = np.arange(0, 10, 0.1)

# params = [
#     {
#         "R": 0.25,
#         "OutPlaceFiringRate" : 5.0,
#         "OutPlaceThetaPhase" : 2.0, #3.14,
#         "InPlacePeakRate" : 15.0,
#         "CenterPlaceField" : 5000.0,
#         "SigmaPlaceField" : 500,
#         "SlopePhasePrecession" : 0.0, #np.deg2rad(10)*10 * 0.001,
#         "ThetaFreq" : 8.0,
#         "PrecessionOnset" : -1.57,
#     },
#     {
#         "R": 0.6,
#         "OutPlaceFiringRate": 0.5,
#         "OutPlaceThetaPhase": 3.14,
#         "InPlacePeakRate": 8.0,
#         "CenterPlaceField": 5000.0,
#         "SigmaPlaceField": 500,
#         "SlopePhasePrecession": 0.0,  # np.deg2rad(10)*10 * 0.001,
#         "ThetaFreq": 8.0,
#         "PrecessionOnset": -1.57,
#     },
# ]
# mask = np.asarray([True, False, False, False, True, False, False, False, False, False])
#
# full_loss_obj = genloss.SpatialThetaGenerators(params, mask)
# full_loss_obj.precomute()
#
# loss = full_loss_obj.get_loss(firings, t)

#####################################################################
# params = [
#     {
#         "R": 0.25,
#         "ThetaFreq": 8.0,
#         "ThetaPhase": 2.0,  # 3.14,
#         "MeanFiringRate" : 5.0,
#     },
#     {
#         "R": 0.5,
#         "ThetaFreq": 8.0,
#         "ThetaPhase": 3.14,
#         "MeanFiringRate" : 4.0,
#     },
# ]
# mask = np.asarray([True, False, False, False, True, False, False, False, False, False])
#
#
# vonmisesloss = genloss.VonMisesLoss(params, mask, sigma_low=0.001)
# vonmisesloss.precomute()
# loss = vonmisesloss.get_loss(firings, t)
#
# print(loss)

#####################################################################
params = [
    {
        "R": 0.25,
        "ThetaFreq": 8.0,
        "LowFiringRateBound" : 0.0,
        "HighFiringRateBound": 50.0,
    },
    {
        "R": 0.5,
        "ThetaFreq": 8.0,
        "LowFiringRateBound" : 0.0,
        "HighFiringRateBound": 50.0,
    },
]
mask = np.asarray([True, False, False, False, True, False, False, False, False, False])


phaselockingloss = genloss.PhaseLockingLoss(params, mask)
phaselockingloss.precomute()
loss = phaselockingloss.get_loss(firings, t)

print(loss)