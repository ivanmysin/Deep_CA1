import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import pickle
import os

os.chdir("../")
from pprint import pprint

import myconfig
from multiprocessing import Pool

AMPL_PYR2PYR_CONNECTIONS = 0.2 # 0.1  # pconn between neubor cells
SIGMA_PYR2PYR_CONNECTIONS = 200 # 50  # mkm

AMPL_PYR2INT_CONNECTIONS = 0.2
SIGMA_PYR2INT_CONNECTIONS = 300 # 90  # mkm

INTsIN_POP = 10  # number cells in population

ALPHA_CA32PYR_PD = -0.005
COEFF_CA32PYR_PD = 0.25

SIGMA_MEC_DV_CONN = 50
MEC_DV_COEFF = 1.2
COEFF_MEC2PYR_PD = 0.2
ALPHA_MEC2PYR_PD = 0.005

ALPHA_LEC2PYR_PD = -0.005

CA3_IN_POP = 375
MEC_IN_POP = 70
LEC_IN_POP = 70



def set_connections(params):
    presyn_pops, all_neuron_populations, CONN_TABLE, INTERNEURONS_TYPES = params

    connections = []
    for pre_idx, presyn in enumerate(presyn_pops):
        for post_idx, postsyn in enumerate(all_neuron_populations):

            if ( ( (presyn["type"] == "CA1 Pyramidal") or
                   (presyn["type"] == "CA1 Pyramidal_generator"))
                    and postsyn["type"] == "CA1 Pyramidal"):
                ## set pyr to pyr connections
                if presyn["z_anat"] * postsyn["z_anat"] < 0:
                    continue

                pretype = "CA1 Pyramidal"

                dist = np.sqrt(
                    (presyn["x_anat"] - postsyn["x_anat"]) ** 2 + (presyn["y_anat"] - postsyn["y_anat"]) ** 2)
                pconn = AMPL_PYR2PYR_CONNECTIONS * np.exp(-0.5 * (dist / SIGMA_PYR2PYR_CONNECTIONS) ** 2)

                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == pretype) & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]

            elif  ( (presyn["type"] == "CA1 Pyramidal") or
                    (presyn["type"] == "CA1 Pyramidal_generator")) and postsyn["type"] in INTERNEURONS_TYPES:
                ## set pyr to int connctions

                pretype = "CA1 Pyramidal"
                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == pretype) & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]

                if len(conn) == 0:
                    continue

                dist = np.sqrt(
                    (presyn["x_anat"] - postsyn["x_anat"]) ** 2 + (presyn["y_anat"] - postsyn["y_anat"]) ** 2)
                pconn = AMPL_PYR2INT_CONNECTIONS * np.exp(-0.5 * (dist / SIGMA_PYR2INT_CONNECTIONS) ** 2)

            elif presyn["type"] in INTERNEURONS_TYPES:  # and postsyn["type"] in INTERNEURONS_TYPES
                ## Set interneurons connections to all

                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == presyn["type"]) & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]

                if len(conn) == 0:
                    continue

                pconn = INTsIN_POP * conn["Connection Probability"].iloc[0]
                # print(pconn)

            elif presyn["type"] == "CA3_generator" and postsyn["type"] == "CA1 Pyramidal":

                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == "CA3 Pyramidal") & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]
                if len(conn) == 0:
                    continue

                if postsyn["z_anat"] > 0:  # for deep pyramidal cells
                    high_level = CA3_IN_POP * conn["Connection Probability"].iloc[0]
                    low_level = COEFF_CA32PYR_PD * high_level
                    pconn = (high_level - low_level) / (1 + np.exp(ALPHA_CA32PYR_PD * postsyn["x_anat"])) + low_level

                else:  # for superficial pyramidal cells
                    pconn = CA3_IN_POP * conn["Connection Probability"].iloc[0]

            elif presyn["type"] == "CA3_generator" and postsyn["type"] in INTERNEURONS_TYPES:
                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == "CA3 Pyramidal") & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]
                if len(conn) == 0:
                    continue

                pconn = CA3_IN_POP * conn["Connection Probability"].iloc[0]

            elif presyn["type"] == "MEC_generator":
                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == "EC LIII Pyramidal") & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]
                if len(conn) == 0:
                    continue

                high_level = MEC_IN_POP * conn["Connection Probability"].iloc[0]
                low_level = COEFF_MEC2PYR_PD * high_level
                pd_axis = (high_level - low_level) / (1 + np.exp(ALPHA_MEC2PYR_PD * postsyn["x_anat"])) + low_level
                dv_axis = np.exp(
                    -0.5 * ((postsyn["y_anat"] - MEC_DV_COEFF * presyn["y_anat"]) / SIGMA_MEC_DV_CONN) ** 2)

                pconn = pd_axis * dv_axis
                if postsyn["z_anat"] < 0:  # for superficial pyramidal cells
                    pconn = pconn * 0.2

            elif presyn["type"] == "LEC_generator":
                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == "EC LIII Pyramidal") & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]
                if len(conn) == 0:
                    continue

                high_level = MEC_IN_POP * conn["Connection Probability"].iloc[0]
                low_level = COEFF_MEC2PYR_PD * high_level
                pd_axis = (high_level - low_level) / (1 + np.exp(ALPHA_LEC2PYR_PD * postsyn["x_anat"])) + low_level
                dv_axis = np.exp(
                    -0.5 * ((postsyn["y_anat"] - MEC_DV_COEFF * presyn["y_anat"]) / SIGMA_MEC_DV_CONN) ** 2)

                pconn = pd_axis * dv_axis
                if postsyn["z_anat"] > 0:  # for deep pyramidal cells
                    pconn = pconn * 0.2

            else:
                continue

            if pconn < myconfig.PCONN_THRESHOLD:
                continue

            if pconn > 1:
                pconn = 1.0

            gsyn_max = conn['g'].values[0]

            conn = {
                "pre_type": presyn["type"],
                "post_type": postsyn["type"],
                "pre_idx": pre_idx,
                "post_idx": post_idx,

                "gsyn_max": gsyn_max,
                "pconn": pconn,

            }

            connections.append(conn)
    return connections

def main():

    conn_path = myconfig.SCRIPTS4PARAMSGENERATION + "DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv"
    CONN_TABLE = pd.read_csv(conn_path, header=0)




    with open(myconfig.STRUCTURESOFNET + "neurons.pickle", mode="br") as file:
        neuron_populations = pickle.load(file)



    INTERNEURONS_TYPES = []
    for pop in neuron_populations:
        if (pop["type"] == "CA1 Pyramidal") or ("generator" in pop["type"]):
            continue
        INTERNEURONS_TYPES.append(pop["type"])
    INTERNEURONS_TYPES = set(INTERNEURONS_TYPES)

    if myconfig.N_THREDS > 1:
        params_map = []
        for i in range(myconfig.N_THREDS):
            params_map.append([neuron_populations[i: -1: myconfig.N_THREDS], neuron_populations, CONN_TABLE, INTERNEURONS_TYPES])

        with Pool(myconfig.N_THREDS) as parallel:
            connections = parallel.map(set_connections, params_map)
            connections = sum(connections, [])  # flat list of connections
    else:
        connections = set_connections([neuron_populations, neuron_populations, CONN_TABLE, INTERNEURONS_TYPES])

    print("Number of connections", len(connections))
    with open( myconfig.STRUCTURESOFNET + "connections.pickle", mode="bw") as file:
        pickle.dump(connections, file)

if __name__ == '__main__':

    main()
