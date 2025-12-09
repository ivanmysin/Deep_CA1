import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
os.chdir('../')

import myconfig


params = {'legend.fontsize': '12',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large',
          }
plt.rcParams.update(params)

with h5py.File(myconfig.OUTPUTSPATH + 'theta_history.h5', mode='r') as dfile:
    loss = dfile['loss'][:]

min_epoche_idx = np.argmin(loss)
print(min_epoche_idx, loss[min_epoche_idx])

fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)

ax.plot(loss, color='k', linewidth=2)

ax.set_xlabel('Эпохи обучения')
ax.set_ylabel('Значение функции потерь')

fig.savefig('./outputs/plots/loss_hist.png')


plt.show()


