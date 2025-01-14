import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.saving import load_model
import h5py
import matplotlib.pyplot as plt
import os
import myconfig

USE_SAVED_MODEL = False
IS_FIT_MODEL = True


def integrate_Erev(Erev, tau_syn, Erest=-60.0, dt=0.1):
    ##E_rest = -60.0  ##!!!

    E_t = np.zeros_like(Erev)
    for idx, E in enumerate(E_t):
        E_inf = Erev[idx]
        if idx == 0:
            E0 = Erest
        else:
            E0 = E_t[idx - 1]

        E_t[idx] = E0 - (E0 - E_inf) * (1 - np.exp(-myconfig.DT / tau_syn[idx]))

    E_t = 1 + E_t / 75.0

    return E_t


def get_dataset(path, train2testratio):



    datafiles = sorted( [file for file in os.listdir(path) if file[-5:] ==".hdf5"] )

    # ##!!
    # datafiles = datafiles[:120]

    if len(datafiles) < 2:
        print(f"Empty folder!!! {path}")
        return None

    Niter_train =  int(train2testratio * len(datafiles))
    Niter_test = int( len(datafiles) - Niter_train)

    print("Files in train:", Niter_train, "Files in test:", Niter_test)

    batch_idx = 0
    for idx, datafile in enumerate(datafiles):
        #range(Niter_train + Niter_test)
        if idx <= Niter_train:
            batch_idx = 0

        filepath = "{path}{file}".format(path=path, file=datafile)


        try:
            with (h5py.File(filepath, mode='r') as h5file):

                firing_rate = h5file["firing_rate"][:].ravel()

                ##!!
                ##firing_rate = np.log(firing_rate + 1.0)

                Erevsyn = h5file["Erevsyn"][:].ravel()
                tau_syn = h5file["tau_syn"][:].ravel()

                E_t = integrate_Erev(Erevsyn, tau_syn, Erest=-60.0, dt=myconfig.DT)



                if idx == 0:
                    N_in_time = h5file["firing_rate"].size # 20000
                    N_in_batch = int(myconfig.BATCH_LEN_PART * N_in_time)

                    Nbatches_train = int(Niter_train * N_in_time / N_in_batch)
                    Nbatches_test = int(Niter_test * N_in_time / N_in_batch)

                    Xtrain = np.zeros((Nbatches_train, N_in_batch, 1), dtype=np.float32)
                    Ytrain = np.zeros((Nbatches_train, N_in_batch, 1), dtype=np.float32)

                    Xtest = np.zeros((Nbatches_test, N_in_batch, 1), dtype=np.float32)
                    Ytest = np.zeros((Nbatches_test, N_in_batch, 1), dtype=np.float32)

                for idx_b in range(0, N_in_time, N_in_batch ):
                    e_idx = idx_b + N_in_batch

                    if idx <= Niter_train:
                        X_tmp = Xtrain
                        Y_tmp = Ytrain
                    else:
                        X_tmp = Xtest
                        Y_tmp = Ytest

                    X_tmp[batch_idx, : , 0] = E_t[idx_b : e_idx]
                    Y_tmp[batch_idx, : , 0] = firing_rate[idx_b : e_idx]

                    batch_idx += 1
        except OSError:
            continue

    return Xtrain, Ytrain, Xtest, Ytest

#######################################################################################
def fit_dl_model_of_population(datapath, targetpath, logfile):
    pop_type = datapath.split("/")[-2]


    # ## !!!!
    # try:
    #     with h5py.File(logfile + pop_type + ".h5", "r") as h5file:
    #         hist_len = h5file['loss'].size
    #
    #     if hist_len > 1000:
    #         print(pop_type, " is already fitted!")
    #         return
    # except FileNotFoundError:
    #     pass


    train2testratio = myconfig.TRAIN2TESTRATIO
    dataset = get_dataset(datapath, train2testratio)
    if dataset is None:
        return

    Xtrain, Ytrain, Xtest, Ytest = dataset

    if USE_SAVED_MODEL:
        model = load_model(targetpath)
    else:

        # create and fit the LSTM network
        model = Sequential()
        model.add( Input(shape=(None, 1)) )
        # model.add( GRU(32, return_sequences=True, kernel_initializer=keras.initializers.Zeros(), \
        #                 stateful=False, recurrent_dropout=0.0, dropout=0.1, \
        #                 kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01),\
        #                 recurrent_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01))) # , stateful=True
        # model.add( GRU(32, return_sequences=True, kernel_initializer=keras.initializers.Zeros(), \
        #                 stateful=False, recurrent_dropout=0.0, dropout=0.1, \
        #                 kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01),\
        #                 recurrent_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01))) # , stateful=True
        #model.add(LSTM(32, return_sequences=True))
        #model.add( Dense(units=16, activation='leaky_relu' ) )  #
        model.add( GRU(units=8, return_sequences=True) )
        #model.add( Dense(units=16, activation='leaky_relu' ) )  #
        model.add( Dense(units=1, activation=keras.ops.square) ) #  'exponential'

        model.compile(loss='log_cosh', optimizer=keras.optimizers.Adam(learning_rate=0.0005), metrics = ['mae', 'mse', 'mean_squared_logarithmic_error'])

    if IS_FIT_MODEL:
        hist = model.fit(Xtrain, Ytrain, epochs=myconfig.NEPOCHES, batch_size=myconfig.BATCHSIZE, verbose=myconfig.VERBOSETRANINGPOPMODELS, validation_data=(Xtest, Ytest))
        model.save(targetpath)

        #print("Training of ", datapath, file=logfile)

        with h5py.File(logfile + pop_type + ".h5", "w") as h5file:
            #print(datapath, "Training Loss =", hist.history['loss'][-1], 'Validation Loss = ', hist.history['val_loss'][-1], file=logfile, flush=True)

            #print(pop_type)
            for key, values in hist.history.items():
                h5file.create_dataset(key, data=values)


    print(pop_type, " is fitted!")



def main():

    logfilepath = myconfig.PRETRANEDMODELS

    for datasetspath in os.listdir(myconfig.DATASETS4POPULATIONMODELS):
        datapath = myconfig.DATASETS4POPULATIONMODELS + datasetspath + "/"
        if not os.path.isdir(datapath):
            continue

        # pop_type = datapath.split("/")[-2]
        # if pop_type != 'CA1 Pyramidal':
        #     continue

        targetpath = myconfig.PRETRANEDMODELS + f"{datasetspath}.keras"
        fit_dl_model_of_population(datapath, targetpath, logfilepath)

if __name__ == '__main__':
    main()

