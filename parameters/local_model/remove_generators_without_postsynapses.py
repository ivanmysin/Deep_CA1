import sys
sys.path.append('../../')
import pickle
import myconfig

import os
os.chdir('../../')


def remove_generators_without_postsynapses():
    with open(myconfig.STRUCTURESOFNET + "neurons.pickle", mode="br") as file:
        neurons = pickle.load(file)


    with open(myconfig.STRUCTURESOFNET + "connections.pickle", mode="br") as file:
        connections = pickle.load(file)



    new_neurons = []
    rem_counter = 0

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
            rem_counter += 1

    with open(myconfig.STRUCTURESOFNET + "neurons.pickle", mode="bw") as file:
       pickle.dump(new_neurons, file)

    print(rem_counter, "Generators are removed")

##########################################################

if __name__ == '__main__':
    remove_generators_without_postsynapses()







