import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pprint import pprint


with open("../presimulation_files/neurons.pickle", mode="br") as file:
    neurons = pickle.load(file)
pop_types = set( [n["type"] for n in neurons] )

with open("../presimulation_files/connections.pickle", mode="br") as file:
    connections = pickle.load(file)

connections = pd.DataFrame.from_dict(connections)

#pprint(pop_types)

for pre_type in ['CA1 Pyramidal', 'CA1 Pyramidal_generator']:    # pop_types:
    for post_type in pop_types:

        conns = connections[ (connections["pre_type"] == pre_type)&(connections["post_type"] == post_type) ]

        if len(conns) == 0:
            print(f'Connections from {pre_type} to {post_type} not found!')
            continue

        dists = []
        for conn_idx, conn in conns.iterrows():
            x_pre = neurons[conn["pre_idx"]]["x_anat"]
            y_pre = neurons[conn["pre_idx"]]["y_anat"]

            x_post = neurons[conn["post_idx"]]["x_anat"]
            y_post = neurons[conn["post_idx"]]["y_anat"]

            dist = np.sqrt(  (x_pre - x_post)**2 + (y_pre - y_post)**2  )

            dists.append(dist)



        fig, axis = plt.subplots(nrows=1)
        axis.set_title(pre_type + " -- " + post_type )
        axis.scatter(dists, conns["pconn"])

        plt.show()

