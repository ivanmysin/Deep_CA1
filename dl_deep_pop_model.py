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

def get_dataset(path, train2testratio):



    datafiles = sorted( [file for file in os.listdir(path) if file[-5:] ==".hdf5"] )

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

                if idx == 0:
                    N_in_time = h5file["gexc"].size # 20000
                    N_in_batch = int(0.25 * N_in_time)

                    Nbatches_train = int(Niter_train * N_in_time / N_in_batch)
                    Nbatches_test = int(Niter_test * N_in_time / N_in_batch)

                    Xtrain = np.zeros((Nbatches_train, N_in_batch, 2), dtype=np.float32)
                    Ytrain = np.zeros((Nbatches_train, N_in_batch, 1), dtype=np.float32)

                    Xtest = np.zeros((Nbatches_test, N_in_batch, 2), dtype=np.float32)
                    Ytest = np.zeros((Nbatches_test, N_in_batch, 1), dtype=np.float32)

                for idx_b in range(0, N_in_time, N_in_batch ):
                    e_idx = idx_b + N_in_batch

                    if idx <= Niter_train:
                        X_tmp = Xtrain
                        Y_tmp = Ytrain

                        # # gexc = h5file["gexc"][idx_b : e_idx]
                        # # ginh = h5file["ginh"][idx_b : e_idx]
                        # Erevsyn = h5file["Erevsyn"][idx_b : e_idx].ravel()   #(gexc*0 + -75*ginh)  / (gexc + ginh)
                        #
                        # #Erevsyn = 2.0*(Erevsyn/75.0 + 1)
                        # Erevsyn = 1 + Erevsyn/75.0
                        #
                        # logtausyn = h5file["tau_syn"][idx_b : e_idx].ravel()
                        #
                        # logtausyn = 2 * np.exp(-myconfig.DT/logtausyn) - 1 # np.log( logtausyn + 1.0 ) #### !!!!
                        # # logtausyn = logtausyn / 10.0
                        # #print(logtausyn.min(), logtausyn.max())
                        #
                        # Xtrain[batch_idx, : , 0] = Erevsyn
                        # Xtrain[batch_idx, : , 1] = logtausyn
                        # Ytrain[batch_idx, : , 0] = h5file["firing_rate"][idx_b : e_idx].ravel() #* 100.0
                    else:
                        X_tmp = Xtest
                        Y_tmp = Ytest

                    Erevsyn = h5file["Erevsyn"][idx_b : e_idx].ravel()
                    Erevsyn = 1 + Erevsyn / 75.0

                    logtausyn = h5file["tau_syn"][idx_b : e_idx].ravel()

                    logtausyn =  np.exp(-myconfig.DT / logtausyn) / np.exp(-0.25)

                    X_tmp[batch_idx, : , 0] = Erevsyn
                    X_tmp[batch_idx, : , 1] = logtausyn

                    firing_rate = h5file["firing_rate"][idx_b : e_idx].ravel() #/ 100
                    Y_tmp[batch_idx, : , 0] = firing_rate  # np.log(firing_rate + 1.0)

                    batch_idx += 1
        except OSError:
            continue

    return Xtrain, Ytrain, Xtest, Ytest

#######################################################################################
def fit_dl_model_of_population(datapath, targetpath, logfile):

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
        model.add( Input(shape=(None, 2)) )
        model.add( LSTM(32, return_sequences=True, kernel_initializer=keras.initializers.GlorotNormal(), stateful=False ) ) # , stateful=True
        model.add( Dense(1, activation='relu') ) #

        # model.add( GRU(16, return_sequences=True, kernel_initializer=keras.initializers.HeUniform(), stateful=True ) ) #, stateful=True
        # model.add( GRU(16, return_sequences=True, kernel_initializer=keras.initializers.HeUniform(), stateful=True ) ) # , stateful=True
        # model.add( Dense(1, activation='relu') ) #

        model.compile(loss="log_cosh", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['mae', 'mse', "mean_squared_logarithmic_error"])
        #model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics = ['mae',])

    if IS_FIT_MODEL:
        hist = model.fit(Xtrain, Ytrain, epochs=myconfig.NEPOCHES, batch_size=myconfig.BATCHSIZE, verbose=myconfig.VERBOSETRANINGPOPMODELS, validation_data=(Xtest, Ytest))
        model.save(targetpath)

        #print("Training of ", datapath, file=logfile)
        pop_type = datapath.split("/")[-2]
        with h5py.File(logfile + pop_type + ".h5", "w") as h5file:
            #print(datapath, "Training Loss =", hist.history['loss'][-1], 'Validation Loss = ', hist.history['val_loss'][-1], file=logfile, flush=True)

            #print(pop_type)
            for key, values in hist.history.items():
                h5file.create_dataset(key, data=values)





def main():

    logfilepath = myconfig.PRETRANEDMODELS #+ "logfitmodels.h5"

    # logfile = open(logfilepath, mode="a")
    # logfile.write("################################################\n")


    for datasetspath in os.listdir(myconfig.DATASETS4POPULATIONMODELS):
        datapath = myconfig.DATASETS4POPULATIONMODELS + datasetspath + "/"
        if not os.path.isdir(datapath):
            continue

        targetpath = myconfig.PRETRANEDMODELS + f"{datasetspath}.keras"
        fit_dl_model_of_population(datapath, targetpath, logfilepath)

    #logfile.close()

if __name__ == '__main__':
    main()

# Y_pred = model.predict(Xtest)
#
# t = np.linspace(0, 0.1*Y_pred.shape[1], Y_pred.shape[1])
# for idx in range(100):
#     fig, axes = plt.subplots(nrows=3)
#
#     axes[0].set_title(idx)
#     axes[0].plot(t, Ytest[idx, :, 0], label="Izhikevich model", color="red")
#     axes[0].plot(t, Y_pred[idx, :, 0], label="LSTM", color="green")
#
#
#     axes[0].legend(loc="upper right")
#
#     axes[1].plot(t, Xtest[idx, :, 0], label="Synaptic Erev", color="orange")
#     axes[2].plot(t, Xtest[idx, :, 1], label="Synaptic tau", color="blue")
#
#     axes[1].legend(loc="upper right")
#     axes[2].legend(loc="upper right")
#
#     plt.show(block=True)
#
#     # if idx > 20:
#     #     break
