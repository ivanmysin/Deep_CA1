import numpy as np
import matplotlib .pyplot as plt

fr = np.linspace(0.0, 150, 10000)
# k1 = 1000
# left_wall = -np.log(k1 * (fr + 0.001) ) / k1
# right_wall = -np.log(k1 * (200 - fr) ) / k1
left_wall = -np.log(1 + np.exp(fr) )


#l = left_wall + right_wall




plt.plot(fr, left_wall)
#plt.plot(fr, l)
plt.show()

