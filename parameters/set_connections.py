import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pickle
import myconfig

def main():

    conn_path = myconfig.SCRIPTS4PARAMSGENERATION + "DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv"
    CONN_TABLE = pd.read_csv(conn_path, header=0)


    AMPL_PYR2PYR_CONNECTIONS = 0.1 # pconn between neubor cells
    SIGMA_PYR2PYR_CONNECTIONS = 50 # mkm

    AMPL_PYR2INT_CONNECTIONS = 0.2
    SIGMA_PYR2INT_CONNECTIONS = 90 # mkm

    INTsIN_POP = 10 # number cells in population

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

    with open(myconfig.STRUCTURESOFNET + "neurons.pickle", mode="br") as file:
        neuron_populations = pickle.load(file)



    INTERNEURONS_TYPES = []
    for pop in neuron_populations:
        if (pop["type"] == "CA1 Pyramidal") or ("generator" in pop["type"]):
            continue
        INTERNEURONS_TYPES.append(pop["type"])
    INTERNEURONS_TYPES = set(INTERNEURONS_TYPES)

    #print(INTERNEURONS_TYPES)

    connections = []
    for pre_idx, presyn in enumerate(neuron_populations):
        for post_idx, postsyn in enumerate(neuron_populations):

            if presyn["type"] == "CA1 Pyramidal" and postsyn["type"] == "CA1 Pyramidal":
                ## set pyr to pyr connections
                if presyn["z_anat"] * postsyn["z_anat"] < 0:
                    continue

                dist = np.sqrt( (presyn["x_anat"] - postsyn["x_anat"])**2 + (presyn["y_anat"] - postsyn["y_anat"])**2 )
                pconn = AMPL_PYR2PYR_CONNECTIONS * np.exp(-0.5 * (dist / SIGMA_PYR2PYR_CONNECTIONS)**2 )

            elif presyn["type"] == "CA1 Pyramidal" and postsyn["type"] in INTERNEURONS_TYPES:
                ## set pyr to int connctions

                dist = np.sqrt((presyn["x_anat"] - postsyn["x_anat"])**2 + (presyn["y_anat"] - postsyn["y_anat"])**2 )
                pconn = AMPL_PYR2INT_CONNECTIONS * np.exp(-0.5 * (dist / SIGMA_PYR2INT_CONNECTIONS)**2 )

            elif presyn["type"] in INTERNEURONS_TYPES: # and postsyn["type"] in INTERNEURONS_TYPES
                ## Set interneurons connections to all
                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == presyn["type"]) & ( \
                                   CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]

                if len(conn) == 0:
                    continue

                pconn = INTsIN_POP * conn["Connection Probability"].iloc[0]
                #print(pconn)

            elif presyn["type"] == "ca3_generator" and postsyn["type"] == "CA1 Pyramidal":

                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == "CA3 Pyramidal") & ( \
                                   CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]
                if len(conn) == 0:
                    continue

                if postsyn["z_anat"] > 0: # for deep pyramidal cells
                    high_level = CA3_IN_POP * conn["Connection Probability"].iloc[0]
                    low_level = COEFF_CA32PYR_PD * high_level
                    pconn = (high_level - low_level) / (1 + np.exp(ALPHA_CA32PYR_PD * postsyn["x_anat"]) ) + low_level

                else: # for superficial pyramidal cells
                    pconn = CA3_IN_POP * conn["Connection Probability"].iloc[0]

            elif presyn["type"] == "ca3_generator" and postsyn["type"] in INTERNEURONS_TYPES:
                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == "CA3 Pyramidal") & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]
                if len(conn) == 0:
                    continue

                pconn = CA3_IN_POP * conn["Connection Probability"].iloc[0]

            elif presyn["type"] == "mec_generator":
                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == "EC LIII Pyramidal") & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]
                if len(conn) == 0:
                    continue

                high_level = MEC_IN_POP * conn["Connection Probability"].iloc[0]
                low_level = COEFF_MEC2PYR_PD * high_level
                pd_axis = (high_level - low_level) / (1 + np.exp(ALPHA_MEC2PYR_PD * postsyn["x_anat"])) + low_level
                dv_axis = np.exp(-0.5 * (  (postsyn["y_anat"] - MEC_DV_COEFF*presyn["y_anat"])/SIGMA_MEC_DV_CONN )**2   )

                pconn = pd_axis * dv_axis
                if postsyn["z_anat"] < 0: # for superficial pyramidal cells
                    pconn = pconn * 0.2

            elif presyn["type"] == "lec_generator":
                conn = CONN_TABLE[(CONN_TABLE["Presynaptic Neuron Type"] == "EC LIII Pyramidal") & ( \
                            CONN_TABLE["Postsynaptic Neuron Type"] == postsyn["type"])]
                if len(conn) == 0:
                    continue

                high_level = MEC_IN_POP * conn["Connection Probability"].iloc[0]
                low_level = COEFF_MEC2PYR_PD * high_level
                pd_axis = (high_level - low_level) / (1 + np.exp(ALPHA_LEC2PYR_PD * postsyn["x_anat"])) + low_level
                dv_axis = np.exp(-0.5 * (  (postsyn["y_anat"] - MEC_DV_COEFF*presyn["y_anat"])/SIGMA_MEC_DV_CONN )**2   )

                pconn = pd_axis * dv_axis
                if postsyn["z_anat"] > 0: # for deep pyramidal cells
                    pconn = pconn * 0.2

            else:
                continue

            if pconn < 10e-4:
                continue

            if pconn > 1:
                pconn = 1.0


            conn = {
                "pre_type" : presyn["type"],
                "post_type" : postsyn["type"],
                "pre_idx" : pre_idx,
                "post_idx" : post_idx,

                "gsyn" : None,
                "pconn" : pconn,

            }

            connections.append(conn)

        print(presyn['type'], " processed!")

    #print(len(connections))
    with open( myconfig.STRUCTURESOFNET + "connections.pickle", mode="bw") as file:
        pickle.dump(connections, file)

if __name__ == '__main__':
    main()
