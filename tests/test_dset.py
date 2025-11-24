import numpy as np
import matplotlib.pyplot as plt
import h5py


dset_file = '../outputs/dataset.h5'

with h5py.File(dset_file, 'r') as f:

    # X = f['Xtrain'][:]
    #
    # Y = f['Y_full_outputs'][:]
    Yphase_lock = f['Y_phase_output'][:]


axo = Yphase_lock[0, :, 41]

axo = axo *100

print(axo)

# phi = np.angle(axo[0] + 1j * axo[1])
phi = np.arctan2(axo[1], axo[0])

print(phi)

# print(X.shape)
# plt.plot(Yphase_lock[10, :, 12])
#
# plt.show()