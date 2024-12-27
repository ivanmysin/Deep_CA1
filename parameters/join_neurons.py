import sys
sys.path.append('../')

import os
import pickle
import myconfig


def main():
    parameters_files = ["pyramidal_cells.pickle", "interneurons.pickle", "CA3_generators.pickle", "MEC_generators.pickle", "LEC_generators.pickle"]
    neurons = []

    for pfile in parameters_files:
        pfile = "_" + pfile

        with open(myconfig.STRUCTURESOFNET + pfile, mode="br") as file:
            neuron = pickle.load(file)
        neurons.extend(neuron)

    #     print(pfile, len(neuron))
    #
    # types = set([pop["type"]  for pop in  neurons])
    #print(types)

    with open(myconfig.STRUCTURESOFNET + "neurons.pickle", mode="bw") as file:
        pickle.dump(neurons, file)


if __name__ == "__main__":
    os.chdir("../")
    main()