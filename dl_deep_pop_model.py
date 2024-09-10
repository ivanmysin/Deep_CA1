import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM
from keras.regularizers import l2
from tensorflow.keras.saving import load_model
import h5py
import matplotlib.pyplot as plt
import os

USE_SAVED_MODEL = False
IS_FIT_MODEL = True

def get_dataset(path, train2testratio):



    datafiles = [file for file in os.listdir(path) if file[-5:] ==".hdf5"]

    Niter_train =  int(train2testratio * len(datafiles))
    Niter_test = int( len(datafiles) - Niter_train)

    print("Files in train:", Niter_train, "Files in test:", Niter_test)

    batch_idx = 0
    for idx, datafile in enumerate(datafiles):
        #range(Niter_train + Niter_test)
        if idx <= Niter_train:
            batch_idx = 0

        filepath = "{path}{file}".format(path=path, file=datafile)
        with h5py.File(filepath, mode='r') as h5file:

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

                    gexc = h5file["gexc"][0, idx_b : e_idx].ravel()
                    ginh = h5file["ginh"][0, idx_b : e_idx].ravel()
                    Erevsyn = (gexc*0 + -75*ginh)  / (gexc + ginh)

                    Erevsyn = 2.0*(Erevsyn/75.0 + 1)

                    logtausyn = np.log( 0.1 / (gexc + ginh + 0.0001) + 1)

                    print(logtausyn.min(), logtausyn.max())

                    Xtrain[batch_idx, : , 0] = Erevsyn # h5file["gexc"][0, idx_b : e_idx].ravel() #/ 80.0
                    Xtrain[batch_idx, : , 1] = logtausyn #/ 50.0

                    #target_firing_rate.append(h5file["firing_rate"][idx_b : e_idx].ravel())

                    Ytrain[batch_idx, : , 0] = h5file["firing_rate"][idx_b : e_idx].ravel() * 100.0
                else:

                    Xtest[batch_idx, : , 0] = h5file["gexc"][0, idx_b : e_idx].ravel() #/ 80.0
                    Xtest[batch_idx, : , 1] = h5file["ginh"][0, idx_b : e_idx].ravel() #/ 50.0
                    Ytest[batch_idx, : , 0] = h5file["firing_rate"][idx_b : e_idx].ravel() * 100.0



                batch_idx += 1

    return Xtrain, Ytrain, Xtest, Ytest

#######################################################################################
datapath = "population_datasets/CA1 Basket/"
train2testratio = 0.7
Xtrain, Ytrain, Xtest, Ytest = get_dataset(datapath, train2testratio)



if USE_SAVED_MODEL:
    model = load_model("pv_bas.keras")
else:

    # create and fit the LSTM network
    model = Sequential()
    # model.add(LSTM(32, input_shape=(None, 2), return_sequences=True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01) ))
    # model.add(LSTM(32, input_shape=(None, 2), return_sequences=True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01) ))
    # model.add(Dense(1, activation='relu'))

    # model.add( Dense(2, activation='relu', kernel_initializer=keras.initializers.HeUniform()) )
    model.add( Input(shape=(None, 2)) )
    model.add( LSTM(32, return_sequences=True, kernel_initializer=keras.initializers.HeUniform() ) )
    model.add( Dense(1, activation='relu') ) #

    model.compile(loss='log_cosh', optimizer=keras.optimizers.Adam(learning_rate=0.0003), metrics = ['mae', 'mean_squared_logarithmic_error'])
    #model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics = ['log_cosh',])

if IS_FIT_MODEL:
    for idx in range(10):
        # model.fit(Xtrain, Ytrain, epochs=20, batch_size=100, verbose=2, validation_data=(Xtest, Ytest))
        model.fit(Xtrain, Ytrain, epochs=20, batch_size=100, verbose=2, validation_data=(Xtest, Ytest))
        model.save("./pretrained_models/pv_bas.keras")
        print(idx+1, " epochs fitted!")


Y_pred = model.predict(Xtest)

t = np.linspace(0, 0.1*Y_pred.shape[1], Y_pred.shape[1])
for idx in range(100):
    fig, axes = plt.subplots(nrows=2)

    axes[0].set_title(idx)
    axes[0].plot(t, Ytest[idx, :, 0], label="Izhikevich model", color="red")
    axes[0].plot(t, Y_pred[idx, :, 0], label="LSTM", color="green")


    axes[0].legend(loc="upper right")

    axes[1].plot(t, Xtest[idx, :, 0], label="Ext conductance", color="orange")
    axes[1].plot(t, Xtest[idx, :, 1], label="Inh conductance", color="blue")

    axes[1].legend(loc="upper right")

    plt.show(block=True)

    if idx > 20:
        break
