import os

from keras.src.ops import dtype

os.chdir("../")
import numpy as np

import pandas as pd
import pickle

import myconfig

with open(myconfig.STRUCTURESOFNET + "_neurons.pickle", "rb") as neurons_file:
    populations = pickle.load(neurons_file)


simple_out_mask = np.zeros(len(populations), dtype='bool')
frequecy_filter_out_mask = np.zeros(len(populations), dtype='bool')
phase_locking_out_mask = np.zeros(len(populations), dtype='bool')

for pop_idx, pop in enumerate(populations):

    if pop["type"] == "CA1 Pyramidal":
        simple_out_mask[pop_idx] = True
        continue

    try:
        #print(type(pop["ThetaPhase"]))
        if np.isnan(pop["ThetaPhase"]):
            phase_locking_out_mask[pop_idx] = True
            #print(pop)
            continue
        else:
            frequecy_filter_out_mask[pop_idx] = True
            continue


    except KeyError:
        continue

