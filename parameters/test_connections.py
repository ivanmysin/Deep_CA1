import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


with open("../presimulation_files/neurons.pickle", mode="br") as file:
    neurons = pickle.load(file)
types = set( [n["type"] for n in neurons] )

with open("../presimulation_files/connections.pickle", mode="br") as file:
    connections = pickle.load(file)

connections = pd.DataFrame.from_dict(connections)

for pre_type in types:
    for post_type in types:

        conns = connections[ (connections["pre_type"] == pre_type)&(connections["post_type"] == post_type) ]

        fig, axis = plt.subplots(nrows=1)
        axis.set_title(pre_type + " -- " + post_type )
        axis.scatter(conns["pre_idx"], conns["pconn"])

        plt.show()
