import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import CosineSimilarity


LossObj = CosineSimilarity(axis=1, name='cosine_similarity')


t = np.arange(0, 120, 0.5)

modulation =  np.cos(t * 1.5 * 2 * np.pi  * 0.001 + np.pi) + 1.0


y_true = np.cos(t * 8 * 2 * np.pi  * 0.001 + 0.5*np.pi) * modulation + 5.0
y_true = y_true.reshape(1,-1, 1)

y_pred = np.cos(t * 8 * 2 * np.pi  * 0.001 + 0.5*np.pi) + 15.0
y_pred = y_pred.reshape(1,-1, 1)

loss = LossObj(y_true, y_pred)

print(np.mean(y_true), np.mean(y_pred))
print(loss)


plt.plot(t, y_true.ravel(), t, y_pred.ravel())
plt.plot(t, modulation)

plt.show()
