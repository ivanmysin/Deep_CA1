import sys
sys.path.append('../../')
import numpy as np
import pickle
import pandas as pd
import os
os.chdir("../../")

import myconfig

THETA_SLOPE_DV = 1.3e-3  # rad / mkm (по оси DV)
THETA_R_SLOPE_DV = -6.25e-05  # 1 / mkm (по оси DV)
THETA_R_0 = 0.35

def main():

    X0 = 0 # proxomo-distal axis
    Y0 = 0 # dorsal axis
    dx = 15
    dy = 15

    LenX = dx * 13
    LenY = dy * 13

    interneurons_types = pd.read_excel( myconfig.SCRIPTS4PARAMSGENERATION + "neurons_parameters.xlsx", sheet_name="local_model", header=0)
    interneurons = []

    for type_idx, cells_pop in interneurons_types.iterrows():
        if str(cells_pop["neurons"]) == "CA1 Pyramidal": continue
        if not cells_pop["is_include"]:
            #print(cells_pop)
            continue

        Npops = cells_pop["Npops"]
        coodinates_x = np.random.uniform(low=X0, high=LenX, size=Npops)
        coodinates_y = np.random.uniform(low=Y0, high=LenY, size=Npops)

        for cell_idx in range(Npops):
            print(cells_pop["neurons"])


            int_cell = {
                "type": cells_pop["neurons"],

                "x_anat": coodinates_x[cell_idx],
                "y_anat": coodinates_y[cell_idx],
                "z_anat": 0,

                "ThetaFreq" : myconfig.ThetaFreq,

                "MeanFiringRate": cells_pop["MeanFiringRate"],  # Хорошо бы сделать лог-нормальное распределение
                "ThetaPhase": THETA_SLOPE_DV * coodinates_y[cell_idx] + cells_pop["MeanFiringRate"],  # DV
                "R": THETA_R_SLOPE_DV * coodinates_y[cell_idx] + THETA_R_0,

                "MinFiringRate": 1.0,
                "MaxFiringRate": 80.0,
            }

            interneurons.append(int_cell)


    with open(myconfig.STRUCTURESOFNET +  "interneurons.pickle", mode="bw") as pklfile:
        pickle.dump(interneurons, pklfile)



if __name__ == "__main__":
    main()