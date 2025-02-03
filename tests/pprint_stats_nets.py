import pickle

##sfile = '../presimulation_files/test_neurons.pickle'
sfile = '../presimulation_files/neurons.pickle'

with open(sfile, mode="br") as file:
    populations = pickle.load(file)

##cfile = '../presimulation_files/test_conns.pickle'
cfile = '../presimulation_files/connections.pickle'
with open(cfile, mode="br") as file:
    conns = pickle.load(file)

types = [pop['type'] for pop in populations]
uniq_types = set(types)

for pop_type in uniq_types:
    print(pop_type, types.count(pop_type))

print("Число популяций =", len(populations))

# for pop_idx, pop in enumerate(populations):
#
#     conns_counter = 0
#     for conn in conns:
#         if conn["post_idx"] != pop_idx: continue
#
#         conns_counter += 1
#
#     print(pop_idx, conns_counter, pop['type'])


# for conn in conns:
#     if conn["post_idx"] != 0: continue
#
#     print(conn["pre_idx"], conn["post_idx"])