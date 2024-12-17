import numpy as np
import matplotlib.pyplot as plt

mu = 50 / 6
std = 0.05 * (1.8e-2 *1000 + 5) / 6


X = np.random.lognormal(mean=np.log(mu), sigma=std, size=1000)

print("Min =", np.min(X))
print("Median =", np.median(X))
print("Max =", np.max(X))

fig, axes = plt.subplots(ncols=2)
axes[0].hist(X, bins=100, density=True)
axes[1].boxplot(X)

plt.show()
