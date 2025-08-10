import pandas as pd


syns_file = '../parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv'

synapses = pd.read_csv(syns_file)
synapses.rename({"g": "gsyn_max", "u": "Uinc", "Connection Probability": "pconn"}, axis=1, inplace=True)

keys = ['Uinc', 'tau_r', 'tau_f', 'tau_d']
for key in keys:
    print(key, synapses[key].min(), synapses[key].max())