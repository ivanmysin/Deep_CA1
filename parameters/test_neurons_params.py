import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open("../presimulation_files/neurons.pickle", mode="br") as file:
    neurons = pickle.load(file)


types = set( [n["type"] for n in neurons] )

neurons = pd.DataFrame.from_dict(neurons)

for type in types:

    neuron_of_type = neurons[neurons["type"] == type]


    fig, axis = plt.subplots(nrows=2)
    axis[0].set_title(type)
    axis[0].scatter(neuron_of_type["x_anat"], neuron_of_type["OutPlaceThetaPhase"])
    axis[1].scatter(neuron_of_type["y_anat"], neuron_of_type["OutPlaceThetaPhase"])
    plt.show()
