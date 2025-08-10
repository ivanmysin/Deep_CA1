import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pprint import pprint


with open("../presimulation_files/neurons.pickle", mode="br") as file:
    neurons = pickle.load(file)


with open("../presimulation_files/connections.pickle", mode="br") as file:
    connections = pickle.load(file)

#connections = pd.DataFrame.from_dict(connections)

new_neurons = []

#pprint(pop_types)

non_connected_generators_indexes = []

for pop_idx, pop in enumerate(neurons):

    if not "_generator" in pop['type']:
        new_neurons.append(pop)
        continue


    is_exist_conn = False
    for conn in connections:

        if conn['pre_idx'] == pop_idx:
            is_exist_conn = True
            break

    if is_exist_conn:
        new_neurons.append(pop)
    else:
        print('Not connected generator ', pop['type'], pop_idx)
        non_connected_generators_indexes.append(pop_idx)

with open("../presimulation_files/neurons.pickle", mode="bw") as file:
   pickle.dump(new_neurons, file)
