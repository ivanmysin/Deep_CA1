import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-25, 25, 1000)

delta = 1
y = delta / (x**2 + delta**2)
y2 = y**2
y2 = y2 / np.max(y2)


plt.plot(x, y, label='y')
plt.plot(x, y2, label='y2')
plt.legend(loc='upper right')
plt.show()