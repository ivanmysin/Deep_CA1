import numpy as np
import matplotlib.pyplot as plt
exp = np.exp

v_avg = np.linspace(-2, 2, 100)
Mgb = 0.27027027027027023
av_nmda = 0.062 * 57.63

U = 1 / (1 + Mgb * exp(-av_nmda * (v_avg - 1.0) ) )


plt.plot(v_avg, U)
plt.show()