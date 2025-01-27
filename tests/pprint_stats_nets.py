import pickle

sfile = '../presimulation_files/neurons.pickle'

with open(sfile, mode="br") as file:
    populations = pickle.load(file)

cfile = '../presimulation_files/connections.pickle'
with open(cfile, mode="br") as file:
    conns = pickle.load(file)

types = [pop['type'] for pop in populations]
uniq_types = set(types)

for pop_type in uniq_types:
    print(pop_type, types.count(pop_type))
