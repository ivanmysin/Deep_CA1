import pickle
import os
os.chdir('../')
import myconfig

MAX_L = 500

neurons_source_path = myconfig.STRUCTURESOFNET + 's_neurons.pickle'
conns_source_path = myconfig.STRUCTURESOFNET + 's_connections.pickle'
with open(neurons_source_path, 'rb') as file:
    populations = pickle.load(file)
with open(conns_source_path, 'rb') as file:
    connections = pickle.load(file)

removed_pops_idxs = []
old_new_idxs = {}
new_pops = []
new_conns = []
n_types = []
for pop_idx, pop in enumerate(populations):
    if (pop['y_anat'] > MAX_L) and (not 'generator' in pop['type']):
        removed_pops_idxs.append(pop_idx)

    else:
        n_types.append(pop['type'])
        new_pops.append(pop)
        old_new_idxs[pop_idx] = len(new_pops) - 1


for conn in connections:
    if (conn['pre_idx'] in removed_pops_idxs) or (conn['post_idx'] in removed_pops_idxs):
        continue
    else:
        conn['pre_idx'] = old_new_idxs[conn['pre_idx']]
        conn['post_idx'] = old_new_idxs[conn['post_idx']]
        new_conns.append(conn)

uniq_types = set(n_types)

for ut in uniq_types:
    utc = n_types.count(ut)

    print(ut, utc)


neurons_target_path = myconfig.STRUCTURESOFNET + 'neurons.pickle'
conns_target_path = myconfig.STRUCTURESOFNET + 'connections.pickle'
with open(neurons_source_path, 'rb') as file:
    populations = pickle.load(file)
with open(conns_source_path, 'rb') as file:
    connections = pickle.load(file)