import os
os.chdir('../')
print(os.getcwd())
import set_connections
import sys
sys.path.append('./local_model')

import ca1_gens
import ca1_pyrs
import ca1_interneurons
import join_neurons
import pickle
import myconfig




def remove_generators_without_postsynapses():
    with open(myconfig.STRUCTURESOFNET + "neurons.pickle", mode="br") as file:
        neurons = pickle.load(file)


    with open(myconfig.STRUCTURESOFNET + "connections.pickle", mode="br") as file:
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

    with open(myconfig.STRUCTURESOFNET + "neurons.pickle", mode="bw") as file:
       pickle.dump(new_neurons, file)
##########################################################

ca1_pyrs.main()
ca1_gens.main()
ca1_interneurons.main()
join_neurons.main()
set_connections.main()
remove_generators_without_postsynapses()
set_connections.main()







