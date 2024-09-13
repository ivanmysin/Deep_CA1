import pickle
import myconfig


def main():
    parameters_files = ["pyramidal_cells.pickle", "interneurons.pickle", "ca3_generators.pickle", "mec_generators.pickle", "lec_generators.pickle"]
    neurons = []

    for pfile in parameters_files:
        with open(myconfig.STRUCTURESOFNET + pfile, mode="br") as file:
            neuron = pickle.load(file)
        neurons.extend(neuron)

    #print(len(neurons))
    with open(myconfig.STRUCTURESOFNET + "neurons.pickle", mode="bw") as file:
        pickle.dump(neurons, file)