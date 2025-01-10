import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import tensorflow as tf
from tensorflow.keras.saving import load_model
import pandas as pd
import sys
sys.path.append('../')
import myconfig
os.chdir('../')


dt = 0.1
E_rest = -60.0

def validate_model(pop_type, path2models, path2dsets, path2saving, train2testratio):
    model = load_model(path2models + pop_type + '.keras', custom_objects={'square':tf.keras.ops.square})

    path = path2dsets + pop_type + '/'
    datafiles = sorted([file for file in os.listdir(path) if file[-5:] == ".hdf5"])
    Niter_train = int(train2testratio * len(datafiles))
    #Niter_test = int(len(datafiles) - Niter_train)

    #datafiles_train = datafiles[:Niter_train]
    datafiles_test = datafiles[Niter_train:]

    for dfile_idx, dfile in enumerate(datafiles_test):
        with h5py.File(path + dfile, "r") as hfile:
            Erev = hfile["Erevsyn"][:]
            tau_syn = hfile["tau_syn"][:]
            firing_rate = hfile["firing_rate"][:]
            firing_rate = firing_rate.reshape(1, -1)


        E_t = np.zeros_like(Erev)
        for idx, E in enumerate(E_t):
            E_inf = Erev[idx]
            if idx == 0:
                E0 = E_rest
            else:
                E0 = E_t[idx - 1]

            E_t[idx] = E0 - (E0 - E_inf) * (1 - np.exp(-dt / tau_syn[idx]))

        if dfile_idx == 0:
            X_test = np.zeros(shape=(len(datafiles_test), E_t.size, 1), dtype=np.float32)
            firing_rates = firing_rate
        else:
            firing_rates = np.append(firing_rates, firing_rate, axis=0)


        X_test[dfile_idx, :, 0] = 1 + E_t / 75.0

    #print("firing_rates.shape ", firing_rates.shape)


    with tf.device('/cpu:0'):
        firing_rate_pred = model.predict(X_test)

        ## firing_rate_pred = -firing_rate_pred + 500 ###!!!!

        loss = tf.keras.losses.logcosh(firing_rate_pred, firing_rate).numpy()
        # print(loss)
        # val_loss.append(float(loss))
        #
        # firing_rate_preds.append(firing_rate_pred)
        # firing_rates.append(firing_rate)
    #print('loss.shape ', loss.shape)
    valsorted_idx = np.argsort(loss)

    #print('valsorted_idx.shape ', valsorted_idx.shape)

    t = np.linspace(0, firing_rate.size*dt, firing_rate.size )

    best_pred = firing_rate_pred[valsorted_idx[0], :]
    median_pred = firing_rate_pred[valsorted_idx[len(datafiles_test)//2], :]
    poor_pred = firing_rate_pred[valsorted_idx[-1], :]

    fr_best = firing_rates[valsorted_idx[0], :]
    fr_median = firing_rates[valsorted_idx[len(datafiles_test)//2], :]
    fr_poor = firing_rates[valsorted_idx[-1], :]

    fig, axes = plt.subplots(nrows=3, figsize=(15, 5))
    fig.suptitle(pop_type, fontsize=14)
    axes[0].set_title('The best approximation')
    axes[0].plot(t, fr_best, color="red", linewidth=3, label="Target")
    axes[0].plot(t, best_pred, color="blue", linewidth=1, label="Predicted")

    axes[1].set_title('The median approximation')
    axes[1].plot(t, fr_median, color="red", linewidth=3, label="Target")
    axes[1].plot(t, median_pred, color="blue", linewidth=1, label="Predicted")

    axes[2].set_title('The worse approximation')
    axes[2].plot(t, fr_poor, color="red", linewidth=3, label="Target")
    axes[2].plot(t, poor_pred, color="blue", linewidth=1, label="Predicted")

    for ax in axes:
        ax.legend(loc='upper right')
        ax.set_xlabel('Time, ms')
        ax.set_ylabel('Firing rate, spikes/sec')
        ax.set_xlim(0, t[-1])

    fig.tight_layout()

    figfile = path2saving + pop_type.replace(" ", "_") + '.png'
    fig.savefig(figfile)
    plt.close(fig)




def main():
    train2testratio = myconfig.TRAIN2TESTRATIO
    path2models = myconfig.PRETRANEDMODELS
    path2dsets = myconfig.DATASETS4POPULATIONMODELS
    path2saving = '/media/sdisk/Deep_CA1/tests/'

    populations = pd.read_excel(myconfig.SCRIPTS4PARAMSGENERATION + "neurons_parameters.xlsx", sheet_name="Sheet2",
                                     header=0)

    populations = populations[populations['is_include'] == 1]
    pop_types = populations["neurons"].to_list()

    for pop_type in pop_types:
        validate_model(pop_type, path2models, path2dsets, path2saving, train2testratio)

        print(pop_type , "is processed!")




main()