import os
os.chdir("../")

print(os.getcwd())
import myconfig
import h5py


for dirpath, dirnames, filenames in os.walk(myconfig.DATASETS4POPULATIONMODELS):
    for filename in filenames:
        pathfile = dirpath + "/" + filename
        if not "hdf5" in pathfile:
            continue

        try:
            h5file = h5py.File(pathfile, mode='r')

            try:
                tmp = h5file["Erevsyn"][:]
                logtausyn = h5file["tau_syn"][:]
                firing_rate = h5file["firing_rate"][:]

            except:
                print(pathfile)

            h5file.close()
        except:
            print(pathfile)




