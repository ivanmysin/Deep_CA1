import pickle
import numpy as np

neurons = [
    {
        "type" : "CA1 Pyramidal",
        "OutPlaceFiringRate": 0.5,  # Хорошо бы сделать лог-нормальное распределение
        "OutPlaceThetaPhase": 3.14 * 0.5,  # DV
        "R": 0.3,

        "InPlacePeakRate": 8.0,  # Хорошо бы сделать лог-нормальное распределение
        "CenterPlaceField": 10,
        "SigmaPlaceField": 30,

        "SlopePhasePrecession": 0.0, #10,  # DV
        "PrecessionOnset": 3.14,

        "ThetaFreq": 7.0,
        "MinFiringRate" : 0.1,
        "MaxFiringRate" : 50.0,

        "I_ext" : 0,
    },
    {
        "type": "CA1 Pyramidal",
        "OutPlaceFiringRate": 0.5,  # Хорошо бы сделать лог-нормальное распределение
        "OutPlaceThetaPhase": 3.14 * 0.5,  # DV
        "R": 0.3,

        "InPlacePeakRate": 8.0,  # Хорошо бы сделать лог-нормальное распределение
        "CenterPlaceField": 10,
        "SigmaPlaceField": 30,

        "SlopePhasePrecession": 0.0, # 10,  # DV
        "PrecessionOnset": 3.14,

        "ThetaFreq": 7.0,
        "MinFiringRate": 0.1,
        "MaxFiringRate": 50.0,
        "I_ext": 0,
    },
    {
        "type": "CA1 Basket",
        "ThetaFreq": 7.0,

        "MeanFiringRate": 15.0,  # Хорошо бы сделать лог-нормальное распределение
        "ThetaPhase": 3.14*0.5,  # DV
        "R": 0.3,
        "MinFiringRate": 1.0,
        "MaxFiringRate": 80.0,
        "I_ext": 0,
    },
    {
        "type": "CA1 Basket CCK+",
        "ThetaFreq": 7.0,

        "MeanFiringRate": 15.0,  # Хорошо бы сделать лог-нормальное распределение
        "ThetaPhase": -3.14 * 0.5,  # DV
        "R": 0.3,
        "MinFiringRate": 1.0,
        "MaxFiringRate": 80.0,
        "I_ext": 200.0,
    },
    {
        "type": "CA1 Oriens-Alveus",
        "ThetaFreq": 7.0,

        "MeanFiringRate": np.nan,  # Хорошо бы сделать лог-нормальное распределение
        "ThetaPhase": np.nan,  # DV
        "R": 0.3,
        "MinFiringRate": 1.0,
        "MaxFiringRate": 80.0,
        "I_ext": 100.0,
    },
    {
        "type": "CA3_generator",
        "OutPlaceFiringRate": 0.5,  # Хорошо бы сделать лог-нормальное распределение
        "OutPlaceThetaPhase": 3.14*0.5,  # DV
        "R": 0.25,

        "InPlacePeakRate": 30,  # Хорошо бы сделать лог-нормальное распределение
        "CenterPlaceField": 2500,
        "SigmaPlaceField": 500,


        "SlopePhasePrecession": 0.0,  # DV
        "PrecessionOnset": 3.14,

        "ThetaFreq": 8.0,


    },
    {
        "type": "MEC_generator",
        "OutPlaceFiringRate": 0.5,  # Хорошо бы сделать лог-нормальное распределение
        "OutPlaceThetaPhase": -3.14 * 0.5,  # DV
        "R": 0.25,

        "InPlacePeakRate": 50.0,  # Хорошо бы сделать лог-нормальное распределение
        "CenterPlaceField": 7000,
        "SigmaPlaceField": 500,

        "SlopePhasePrecession": 0.0,  # DV
        "PrecessionOnset": 3.14,

        "ThetaFreq": 8.0,
    },
]

connections = [
    # {
    #     "pconn" : 0.0,
    #     "pre_idx" : 0,
    #     "post_idx" : 0,
    #     "gsyn_max": 0.0,
    # },
    # {
    #     "pconn": 1.0,
    #     "pre_idx": 0,
    #     "post_idx": 1,
    #     "gsyn_max": 500.0,
    # },
    # {
    #     "pconn": 0.0,
    #     "pre_idx": 1,
    #     "post_idx": 1,
    #     "gsyn_max": 0.0,
    # },
    # {
    #     "pconn": 0.0,
    #     "pre_idx": 1,
    #     "post_idx": 0,
    #     "gsyn_max": 0.0,
    # },
    #
    # {
    #     "pconn": 0.0,
    #     "pre_idx": 0,
    #     "post_idx": 2,
    #     "gsyn_max": 0.0,
    # },
    # {
    #     "pconn": 0.0,
    #     "pre_idx": 0,
    #     "post_idx": 3,
    #     "gsyn_max": 0.0,
    # },
    {
        "pconn": 1.0,
        "pre_idx": 2,
        "post_idx": 3,
        "gsyn_max": 10.0,
    },

    # {
    #     "pconn": 0.0,
    #     "pre_idx": 5,
    #     "post_idx": 0,
    #     "gsyn_max": 0.0,
    # },
    {
        "pconn": 1.0,
        "pre_idx": 5,
        "post_idx": 1,
        "gsyn_max": 500.0,
    },
    {
        "pconn": 1.0,
        "pre_idx": 5,
        "post_idx": 2,
        "gsyn_max": 50.0,
    },
    # {
    #     "pconn": 0.0,
    #     "pre_idx": 5,
    #     "post_idx": 3,
    #     "gsyn_max": 0.0,
    # },

    {
        "pconn": 1.0,
        "pre_idx": 6,
        "post_idx": 0,
        "gsyn_max": 50.0,
    },
    # {
    #     "pconn": 0.0,
    #     "pre_idx": 6,
    #     "post_idx": 1,
    #     "gsyn_max": 0.0,
    # },
    # {
    #     "pconn": 0.0,
    #     "pre_idx": 6,
    #     "post_idx": 2,
    #     "gsyn_max": 0.0,
    # },
    # {
    #     "pconn": 0.0,
    #     "pre_idx": 6,
    #     "post_idx": 3,
    #     "gsyn_max": 0.0,
    # },
    # {
    #     "pconn": 0.0,
    #     "pre_idx": 6,
    #     "post_idx": 4,
    #     "gsyn_max": 0.0,
    # },
]


for conn in connections:
    conn['pre_type'] = neurons[conn['pre_idx']]['type']
    conn['post_type'] = neurons[conn['post_idx']]['type']

with open("../presimulation_files/test_neurons.pickle", mode="bw") as file:
    pickle.dump(neurons, file)

with open("../presimulation_files/test_conns.pickle", mode="bw") as file:
    pickle.dump(connections, file)