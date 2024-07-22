import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open("../presimulation_files/neurons.pickle", mode="br") as file:
    neurons = pickle.load(file)


types = set( [n["type"] for n in neurons] )

neuron_of_type = pd.DataFrame.from_dict(neurons)

#
# for type in types:
#
#     neuron_of_type = None
#
#     for neuron in neurons:
#         if neuron["type"] == type:
#
#             if neuron_of_type is None:
#                 neuron_of_type = pd.DataFrame.from_dict(neuron)
#             else:
#                 neuron = pd.Series(neuron)
#                 neuron_of_type = pd.concat([neuron_of_type, neuron], ignore_index=True, sort=False)


plt.scatter(neuron_of_type["x_anat"], neuron_of_type["OutPlaceFiringRate"])
plt.show()
