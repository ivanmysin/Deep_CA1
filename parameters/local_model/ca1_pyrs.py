import sys
sys.path.append('../../')
import numpy as np
import pickle

import os
os.chdir("../../")

import myconfig

# Hyperparameters
OUTPLACE_FIRINGRATE_GEN = 'lognormal'  # 'normal' or 'constant'
INPLACE_FIRINGRATE_GEN = 'lognormal'  # 'normal' or 'constant'
PLACESIZE_GEN = 'lognormal'  # 'normal' or 'constant'

TRACK_LENGTH = 250 # 400  # cm
PLACECELLSPROB = 0.5  # Вероятность пирамидного нейрона стать клеткой места в одном лабиринте
PHASEPRECPROB = 0.5  # Вероятность обнаружить фазовую прецессию у клетки места
PLACESIZE_MEAN = 20  # см, Средний размер поля места в дорсальном гиппокампе.
PLACESIZE_STD = 5  # см, Стандартное отклонение размера поля места в дорсальном гиппокампе.

PEAKFIRING = 8.0
PEAKFIRING_STD = 0.4
OUTPLACEFIRING = 0.5
OUTPLACEFIRING_STD = 0.7

PLACESIZE_SLOPE_DV = 1.8e-2  # cm (поля места) / mkm (по оси DV)
THETA_SLOPE_DV = 1.3e-3  # rad / mkm (по оси DV)
THETA_R_SLOPE_DV = -6.25e-05  # 1 / mkm (по оси DV)
THETA_R_0 = 0.25

PHASEPREC_SLOPE_DEEP_0 = 6.0
PHASEPREC_SLOPE_SUP_0 = 4.0

PHASEPREC_ONSET_DEEP_0 = 2.79
PHASEPREC_ONSET_SUP_0 = 3.84

PHASEPREC_SLOPE_DECREASE_DV = -1.8e-03  # rad / mkm (по оси DV)

THETA_LOCALPHASE_DEEP = 0.0  # rad
THETA_LOCALPHASE_SUP = np.pi  # rad

def get_grid(Nx, Ny, dx, dy, x0, y0):

    x1 = x0 + Nx*dx
    y1 = y0 + Ny*dy

    # x = np.linspace(x0, x1, Nx)
    # y = np.linspace(y0, y1, Ny)

    x = np.arange(x0, x1, dx)
    y = np.arange(y0, y1, dy)

    xv, yv = np.meshgrid(x, y)

    return xv, yv


def make_ca1_pyrs_coords(X0, Y0, dx, dy, Npyrs_sim_x, Npyrs_sim_y, Nbottom_gens_x, Nbottom_gens_y, Nleft_gens_x, Nleft_gens_y):

    x0 = X0 + dx * Nleft_gens_x
    y0 = Y0 + dy * Nbottom_gens_y



    xv_sim, yv_sim = get_grid(Npyrs_sim_x, Npyrs_sim_y, dx, dy, x0, y0)


    xv_gens_bottom, yv_gens_bottom = get_grid(Nbottom_gens_x, Nbottom_gens_y, dx, dy, X0, Y0)

    yv_gens_top = yv_gens_bottom + dy*(Npyrs_sim_y + Nbottom_gens_y)
    xv_gens_top = np.copy(xv_gens_bottom)

    x0_left = X0
    y_left = y0
    xv_gens_left, yv_gens_left = get_grid(Nleft_gens_x, Nleft_gens_y, dx, dy, x0_left, y_left)

    yv_gens_right = np.copy(yv_gens_left)
    xv_gens_right = xv_gens_left + dx*(Nleft_gens_x + Npyrs_sim_x)

    xv_gens = np.concatenate( [xv_gens_left.ravel(), xv_gens_top.ravel(),xv_gens_right.ravel(), xv_gens_bottom.ravel()]  )
    yv_gens = np.concatenate([yv_gens_left.ravel(), yv_gens_top.ravel(), yv_gens_right.ravel(), yv_gens_bottom.ravel()])

    xv_sim = xv_sim.ravel()
    yv_sim = yv_sim.ravel()

    return xv_sim, yv_sim, xv_gens, yv_gens


def get_cells_list(pyr_coodinates_x, pyr_coodinates_y, pyr_coodinates_z, ThetaPhase, preces_slope0, precess_onset0, is_gen=False):

    pyramidal_cells = []

    for pyrs_x, pyrs_y, pyrs_z in zip(pyr_coodinates_x, pyr_coodinates_y, pyr_coodinates_z):

        if PLACECELLSPROB < np.random.rand():
            center_place_field = np.random.uniform(low=0.0, high=TRACK_LENGTH, size=1)[0]
        else:
            center_place_field = -1000000

        if PHASEPRECPROB < np.random.rand() and center_place_field > 0:
            phase_precession_slope = PHASEPREC_SLOPE_DECREASE_DV * pyrs_y + preces_slope0
        else:
            phase_precession_slope = 0.0

        place_size = (PLACESIZE_SLOPE_DV * pyrs_y + PLACESIZE_MEAN) / 6
        place_size_std = (PLACESIZE_SLOPE_DV * pyrs_y + PLACESIZE_STD) / 6

        if PLACESIZE_GEN == 'lognormal':
            place_size = np.random.lognormal(mean=np.log(place_size), sigma=0.05 * place_size_std)  # !!!!!!

        elif PLACESIZE_GEN == 'normal':
            place_size = np.random.normal(loc=place_size, scale=place_size_std)

        outplacefiringrate = OUTPLACEFIRING
        if OUTPLACE_FIRINGRATE_GEN == 'lognormal':
            outplacefiringrate = np.random.lognormal(mean=np.log(outplacefiringrate), sigma=OUTPLACEFIRING_STD)
        elif OUTPLACE_FIRINGRATE_GEN == 'normal':
            outplacefiringrate = np.random.normal(loc=outplacefiringrate, scale=OUTPLACEFIRING_STD)

        inplacefiringrate = PEAKFIRING
        if INPLACE_FIRINGRATE_GEN == 'lognormal':
            inplacefiringrate = np.random.lognormal(mean=np.log(inplacefiringrate), sigma=PEAKFIRING_STD)
        elif INPLACE_FIRINGRATE_GEN == 'normal':
            inplacefiringrate = np.random.normal(loc=inplacefiringrate, scale=PEAKFIRING_STD)

        if is_gen:
            cell_type =  "CA1 Pyramidal_generator"
        else:
            cell_type = "CA1 Pyramidal"

        pyr_cell = {
            "type": cell_type,

            "x_anat": pyrs_x,
            "y_anat": pyrs_y,
            "z_anat": pyrs_z,

            "ThetaFreq": myconfig.ThetaFreq,

            "OutPlaceFiringRate": outplacefiringrate,
            "OutPlaceThetaPhase": THETA_SLOPE_DV * pyrs_y + ThetaPhase,  # DV
            "R": THETA_R_SLOPE_DV * pyrs_y + THETA_R_0,

            "InPlacePeakRate": inplacefiringrate,
            "CenterPlaceField": float(center_place_field),
            "SigmaPlaceField": place_size,

            "SlopePhasePrecession": phase_precession_slope,  # DV
            "PrecessionOnset": THETA_SLOPE_DV * pyrs_y + precess_onset0,

            "MinFiringRate": 0.1,
            "MaxFiringRate": 50.0,

        }

        pyramidal_cells.append(pyr_cell)

    return pyramidal_cells


def main():
    X0 = 0 # proxomo-distal axis
    Y0 = 0 # dorsal axis
    dx = 15
    dy = 15
    Nbottom_gens_x = 13
    Nbottom_gens_y = 3

    Npyrs_sim_x = 7
    Npyrs_sim_y = 7

    Nleft_gens_x = 3
    Nleft_gens_y = Npyrs_sim_y


    xv_sim, yv_sim, xv_gens, yv_gens = make_ca1_pyrs_coords(X0, Y0, dx, dy, Npyrs_sim_x, Npyrs_sim_y, Nbottom_gens_x,
                                                            Nbottom_gens_y, Nleft_gens_x, Nleft_gens_y)
    pyr_coodinates_x = xv_sim
    pyr_coodinates_y = yv_sim

    pyramidal_cells = []
    pyramidal_cells_gens = []

    for radial_axis in ["deep", "sup"]:
        for sim_type in ["sim", "gen"]:

            if radial_axis == "deep":

                pyr_coodinates_z = np.zeros_like(yv_sim) + 1.0

                ThetaPhase = THETA_LOCALPHASE_DEEP
                preces_slope0 = PHASEPREC_SLOPE_DEEP_0
                precess_onset0 = PHASEPREC_ONSET_DEEP_0


            elif radial_axis == "sup":
                pyr_coodinates_z = np.zeros_like(yv_sim) - 1.0
                ThetaPhase = THETA_LOCALPHASE_SUP
                preces_slope0 = PHASEPREC_SLOPE_SUP_0
                precess_onset0 = PHASEPREC_ONSET_SUP_0

            if sim_type == 'sim':
                is_gen = False
            elif sim_type == 'gen':
                is_gen = True

            pyramidal_cells_tmp = get_cells_list(pyr_coodinates_x, pyr_coodinates_y, pyr_coodinates_z, ThetaPhase, preces_slope0, precess_onset0, is_gen=is_gen)

            if sim_type == 'sim':
                pyramidal_cells.extend(pyramidal_cells_tmp)

            elif sim_type == 'gen':
                pyramidal_cells_gens.extend(pyramidal_cells_tmp)

    with open(myconfig.STRUCTURESOFNET + "pyramidal_cells.pickle", mode="bw") as file:
        pickle.dump(pyramidal_cells, file)

    with open(myconfig.STRUCTURESOFNET +  "ca1_pyramidal_cells_generators.pickle", mode="bw") as file:
        pickle.dump(pyramidal_cells_gens, file)


    return pyr_coodinates_x, pyr_coodinates_y, pyr_coodinates_z


if __name__ == "__main__":

    pyr_coodinates_x, pyr_coodinates_y, pyr_coodinates_z = main()
