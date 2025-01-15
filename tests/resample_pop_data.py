import numpy as np
from scipy.signal import resample
import h5py
import os

sdt = 0.1 # source dt
tdt = 0.5 # target dt
dsf = sdt / tdt # downsampling factor
NN = 4000
duration = 2000

source_path =  '/home/ivan/PycharmProjects/Deep_CA1/population_datasets/' # '/media/sdisk/Deep_CA1/population_datasets/'
target_path = '/media/sdisk/Deep_CA1/new_population_datasets/'

for dirpath, _, filenames in os.walk(source_path):

    if len(filenames) > 0:
        pop_type = dirpath.split("/")[-1]

        if not 'CA1' in pop_type:
            continue
        saving_path = target_path + pop_type + '/'


        # if not os.path.isdir(saving_path):
        #     os.mkdir(saving_path)
    else:
        continue


    for filename in filenames:
        source_file = os.path.join(dirpath, filename)
        target_file = os.path.join(saving_path, filename)

        sh5file = h5py.File(source_file, mode='r')
        th5file = h5py.File(target_file, mode='w')

        ginh = sh5file['ginh'][:].ravel()
        gexc = sh5file['gexc'][:].ravel()

        ginh = resample(ginh, int(ginh.size * dsf) )
        gexc = resample(gexc, int(gexc.size * dsf) )


        Erevsyn = sh5file['Erevsyn'][:].ravel()
        tau_syn = sh5file['tau_syn'][:].ravel()

        Erevsyn = resample(Erevsyn, int(Erevsyn.size * dsf) )
        tau_syn = resample(tau_syn, int(tau_syn.size * dsf) )

        firing_t = sh5file['firing_t'][:].ravel()
        firing_i = sh5file['firing_i'][:].ravel()

        firing_rate, bins = np.histogram(firing_t, range=[0, duration], bins=int(duration / tdt) )
        dbins = bins[1] - bins[0]
        firing_rate = firing_rate / NN / (0.001 * dbins)


        th5file.create_dataset('firing_i', data=firing_i)
        th5file.create_dataset('firing_t', data=firing_t)
        th5file.create_dataset('firing_rate', data=firing_rate.astype(np.float32).ravel())
        th5file.create_dataset('Erevsyn', data=Erevsyn.astype(np.float32).ravel())
        th5file.create_dataset('tau_syn', data=tau_syn.astype(np.float32).ravel())
        th5file.attrs['dt'] = dbins

        th5file.create_dataset('gexc', data=gexc)
        th5file.create_dataset('ginh', data=ginh)

        sh5file.close()
        th5file.close()
        print(source_file)
        print(target_file)
        print('######################################')

