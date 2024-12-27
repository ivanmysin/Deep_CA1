import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import os
os.chdir("../")
import myconfig

def main():
    TRACK_LENGTH = 400  # cm
    THETA_FREQ = myconfig.ThetaFreq # Hz


    for INPUT_TYPE in ["MEC", "LEC", "CA3"]:

        # Hyperparameters
        if INPUT_TYPE == "MEC":
            Npyr = 200 ###!!!!!!!!! 200

            PLACESIZE_MEAN = 20  # см, Средний размер поля места в дорсальном гиппокампе.
            PLACESIZE_STD = 5  # см, Стандартное отклонение размера поля места в дорсальном гиппокампе.

            PEAKFIRING = 8.0
            OUTPLACEFIRING = 1.5

            PLACESIZE_SLOPE_DV = 3.48e-3 # cm (поля места) / mkm (по оси DV)
            THETA_SLOPE_DV = 0.45e-3 # rad / mkm (по оси DV)
            THETA_R_SLOPE_DV = -6.25e-05 # 1 / mkm (по оси DV)
            THETA_R_0 = 0.25

            THETA_PHASE_0 = -0.52
            DV_LENGTH = 2000 # mkm

            PHASEPREC_SLOPE = 2
            PHASEPREC_SLOPE_DECREASE_DV = -2.82e-05  # rad / mkm (по оси DV)
            PHASEPREC_ONSET = THETA_PHASE_0 - 0.5
            PHASEPRECPROB = 0.5  # Вероятность обнаружить фазовую прецессию у клетки места

        elif INPUT_TYPE == "LEC":
            Npyr = 200

            PLACESIZE_MEAN = 30  # см, Средний размер поля места в дорсальном гиппокампе.
            PLACESIZE_STD = 10  # см, Стандартное отклонение размера поля места в дорсальном гиппокампе.

            PEAKFIRING = 8.0
            OUTPLACEFIRING = 1.5

            PLACESIZE_SLOPE_DV = 3.48e-3 # cm (поля места) / mkm (по оси DV)
            THETA_SLOPE_DV = 0.45e-3 # rad / mkm (по оси DV)
            THETA_R_SLOPE_DV = -6.25e-05 # 1 / mkm (по оси DV)
            THETA_R_0 = 0.15

            THETA_PHASE_0 = -0.52
            DV_LENGTH = 2000 # mkm

            PHASEPREC_SLOPE = 0.0
            PHASEPREC_SLOPE_DECREASE_DV = 0.0 # rad / mkm (по оси DV)
            PHASEPREC_ONSET = THETA_PHASE_0 - 0.5
            PHASEPRECPROB = 0.0  # Вероятность обнаружить фазовую прецессию у клетки места

        elif INPUT_TYPE == "CA3":
            Npyr = 200

            PLACESIZE_MEAN = 20  # см, Средний размер поля места в дорсальном гиппокампе.
            PLACESIZE_STD = 5  # см, Стандартное отклонение размера поля места в дорсальном гиппокампе.

            PEAKFIRING = 8.0
            OUTPLACEFIRING = 0.5

            PLACESIZE_SLOPE_DV = 1.8e-2  # cm (поля места) / mkm (по оси DV)
            THETA_SLOPE_DV = 1.3e-3  # rad / mkm (по оси DV)
            THETA_R_SLOPE_DV = -6.25e-05  # 1 / mkm (по оси DV)
            THETA_R_0 = 0.35

            THETA_PHASE_0 = 1.4
            DV_LENGTH = 2400 # mkm

            PHASEPREC_SLOPE = 5.0
            PHASEPREC_SLOPE_DECREASE_DV = -1.8e-03  # rad / mkm (по оси DV)
            PHASEPREC_ONSET = THETA_PHASE_0 - 0.5
            PHASEPRECPROB = 0.5  # Вероятность обнаружить фазовую прецессию у клетки места


        generators = []
        Nsteps = int( Npyr / (DV_LENGTH+100) / 100) + 1
        for cell_pos_dv in range(0, DV_LENGTH+100, 100):

            place_size = (PLACESIZE_SLOPE_DV * cell_pos_dv + PLACESIZE_MEAN) / 6
            place_size_std = (PLACESIZE_SLOPE_DV * cell_pos_dv + PLACESIZE_STD) / 6

            center_place_field = np.random.uniform(low=0.0, high=TRACK_LENGTH, size=1).ravel()
            if PHASEPRECPROB < np.random.uniform():
                phase_precession_slope = PHASEPREC_SLOPE + cell_pos_dv*PHASEPREC_SLOPE_DECREASE_DV
            else:
                phase_precession_slope = 0.0

            for idx in range(Nsteps):
                gencell = {
                    "type" : f"{INPUT_TYPE}_generator",

                    "x_anat": 0,
                    "y_anat": cell_pos_dv,
                    "z_anat": 0,

                    "OutPlaceFiringRate": OUTPLACEFIRING,  # Хорошо бы сделать лог-нормальное распределение
                    "OutPlaceThetaPhase": THETA_SLOPE_DV * cell_pos_dv + THETA_PHASE_0,  # DV
                    "R": THETA_R_SLOPE_DV * cell_pos_dv + THETA_R_0,

                    "InPlacePeakRate": PEAKFIRING,  # Хорошо бы сделать лог-нормальное распределение
                    "CenterPlaceField": center_place_field,
                    "SigmaPlaceField": np.random.normal(loc=place_size, scale=place_size_std),  # !!!!  Хорошо бы сделать лог-нормальное распределение

                    "SlopePhasePrecession": phase_precession_slope,  # DV
                    "PrecessionOnset": THETA_SLOPE_DV * cell_pos_dv + PHASEPREC_ONSET,

                    "ThetaFreq" : THETA_FREQ,


                }

                generators.append(gencell)

        with open(myconfig.STRUCTURESOFNET + f"{INPUT_TYPE}_generators.pickle", mode="bw") as file:
            pickle.dump(generators, file)

if __name__ == "__main__":

    main()