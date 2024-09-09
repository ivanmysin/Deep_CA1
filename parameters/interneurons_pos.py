import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
import pickle

THETA_SLOPE_DV = 1.3e-3 # rad / mkm (по оси DV)
THETA_R_SLOPE_DV = -6.25e-05 # 1 / mkm (по оси DV)
THETA_R_0 = 0.35


filepath = "CA1_anatomy.csv"
CA1_flat = pd.read_csv(filepath, header=0)
StepH = CA1_flat["H"][1] - CA1_flat["H"][0]

Square_CA1 = StepH * CA1_flat["L"].sum()
print("Square of CA1 field =", Square_CA1, "mkm^2")

right_bound = 0.5*CA1_flat["L"]
left_bound = -0.5*CA1_flat["L"]

coodinates_x = np.empty(shape=0, dtype=np.float64)
coodinates_y = np.empty_like(coodinates_x)
StepProxDist = 100

Npops = 30
stepXY = int( Square_CA1 / (StepProxDist**2 * Npops) )

for slice_idx, l in enumerate(CA1_flat["L"]):
    lb = left_bound[slice_idx] + 0.5*StepProxDist
    rb = right_bound[slice_idx] - 0.5*StepProxDist

    tmp_x = np.arange(lb, rb, StepProxDist)
    coodinates_x = np.append(coodinates_x, tmp_x)
    coodinates_y = np.append(coodinates_y, np.zeros_like(tmp_x) + CA1_flat["H"][slice_idx])

points = np.stack([coodinates_x, coodinates_y]).transpose()


interneurons_types = pd.read_excel("neurons_parameters.xlsx", sheet_name="Sheet2", header=0)
interneurons = []

for type_idx, cells_pop in interneurons_types.iterrows():
    if not cells_pop["is_include"]: continue

    print(cells_pop["Interneurons"])

    selected, _ = kmeans(points, cells_pop["Npops"])

    for cell_idx in range(cells_pop["Npops"]):
        int_cell = {
            "type": cells_pop["Interneurons"],

            "x_anat": selected[cell_idx, 0],
            "y_anat": selected[cell_idx, 1],
            "z_anat": 0,

            "MeanFiringRate": cells_pop["MeanFiringRate"],  # Хорошо бы сделать лог-нормальное распределение
            "ThetaPhase": THETA_SLOPE_DV * selected[cell_idx, 1] + cells_pop["MeanFiringRate"],  # DV
            "R": THETA_R_SLOPE_DV * selected[cell_idx, 1] + THETA_R_0,
        }

        interneurons.append(int_cell)


    fig, axes = plt.subplots()
    axes.plot( right_bound, CA1_flat["H"], color="blue")
    axes.plot( left_bound, CA1_flat["H"], color="blue")

    #axes.scatter(coodinates_x, coodinates_y, s=20, color='blue')
    axes.scatter(points[:, 0], points[:, 1], s=20, color='blue')
    axes.scatter(selected[:, 0], selected[:, 1], s=15, color='red')

    plt.show()



with open("../presimulation_files/interneurons.pickle", mode="bw") as file:
    pickle.dump(interneurons, file)