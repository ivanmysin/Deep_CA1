import numpy as np
import matplotlib.pyplot as plt

M = 1
s = 100

Ebias = 100
#th = 60

E = np.linspace(0, 120, 1000) # +90

x = E + Ebias

nu = M * x**2 / (x**2 + s)
nu[x < 0] = 0

plt.scatter(E-90, nu)
plt.show()


