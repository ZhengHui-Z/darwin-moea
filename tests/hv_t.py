import numpy as np

p = np.array([[1, 1]])
a = np.array([
    [0.7, 0.3],
    [0.5, 0.5],
    [0.2, 0.2]
])

b = np.lexsort(np.rot90(a))

hv = np.sum(a[b, 0] * a[b, 1][::-1])
