import pickle

with open("../presimulation_files/test_conns__.pickle", mode="br") as file:
    connections__ = pickle.load(file)

with open("../presimulation_files/test_conns.pickle", mode="br") as file:
    connections = pickle.load(file)


for conn__, conn in zip(connections__, connections):

    check = (conn__["pre_idx"] == conn["pre_idx"]) and (conn__["post_idx"] == conn["post_idx"]) and (conn__["gsyn_max"] == conn["gsyn_max"])

    if not check:
        print("Not equal!")