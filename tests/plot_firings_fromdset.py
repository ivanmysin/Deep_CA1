import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import myconfig
import os

os.chdir("../")


fir_file = 'firings_19.h5'
populations = pd.read_excel(myconfig.FIRINGSNEURONPARAMS, sheet_name='theta_model')
populations.rename( {'neurons' : 'type'}, axis=1, inplace=True)
populations = populations[populations['Npops'] > 0]


with h5py.File(myconfig.OUTPUTSPATH_FIRINGS + fir_file, mode='r') as dfile:
    firings = dfile['firings'][:]

with h5py.File(myconfig.OUTPUTSPATH + 'dataset.h5', mode='r') as dfile:
    X = dfile['Xtrain'][:]
    Y = dfile['Ytrain'][:]

    t = X.ravel()
    target_firing = Y.reshape( firings.shape  )



for f_idx, pop_type in enumerate(populations['type']):

    if f_idx == firings.shape[-1]: break

    fig, axes = plt.subplots(nrows=1, sharex=True, sharey=False)

    axes.set_title(pop_type)

    axes.plot(t, firings[0, :, f_idx], color='blue')
    axes.plot(t, target_firing[0, :, f_idx], color='red')

    #print(f_idx, np.sum( np.isnan(firings[0, :, f_idx]) ) )


    # if f_idx < targents_pyrs.shape[-1]:
    #     axes[1].plot(t, targents_pyrs[0, :, f_idx], color='red')

    #fig.savefig(myconfig.OUTPUTSPATH_PLOTS + f'{f_idx}.png')

    #plt.close(fig)
plt.show()