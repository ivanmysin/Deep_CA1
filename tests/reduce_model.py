import pickle
import os
os.chdir('../')
import myconfig

MAX_L = 1700

neurons_source_path = myconfig.STRUCTURESOFNET + 'neurons.pickle'
conns_source_path = myconfig.STRUCTURESOFNET + 'connections.pickle'
with open(neurons_source_path, 'rb') as file:
    populations = pickle.load(file)
with open(conns_source_path, 'rb') as file:
    connections = pickle.load(file)

removed_pops_idxs = []
old_new_idxs = {}
new_pops = []
new_conns = []
for pop_idx, pop in enumerate(populations):
    if (pop['y_anat'] > MAX_L) and (not 'generator' in pop['type']):
        removed_pops_idxs.append(pop_idx)

    else:
        new_pops.append(pop)
        old_new_idxs[pop_idx] = len(new_pops) - 1


for conn in connections:
    if (conn['pre_idx'] in removed_pops_idxs) or (conn['post_idx'] in removed_pops_idxs):
        continue
    else:
        conn['pre_idx'] = old_new_idxs[conn['pre_idx']]
        conn['post_idx'] = old_new_idxs[conn['post_idx']]
        new_conns.append(conn)


neurons_target_path = myconfig.STRUCTURESOFNET + 'test_neurons.pickle'
conns_target_path = myconfig.STRUCTURESOFNET + 'test_conns.pickle'
with open(neurons_source_path, 'rb') as file:
    populations = pickle.load(file)
with open(conns_source_path, 'rb') as file:
    connections = pickle.load(file)