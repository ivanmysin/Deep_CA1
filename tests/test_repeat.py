import numpy as np

#a = np.zeros([1, 2, 3], dtype=np.float32)
# a = np.random.rand(1, 2, 3)
#
# a[0, 0, :] += 1
#
# print(a.shape)
#
# b = np.tile(a, (5, 1, 1))
# print(b.shape)
#
# print("="*20)
# print(a[0, :, :])
# print(b[0, :, :])
# print(b[1, :, :])

# a = np.arange(15, 115, 5)
a = np.arange(12, 112, 5)
np.random.shuffle(a)

for i in a:
    print(i)