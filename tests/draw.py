import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
plt.ion()

# for i in range(10):
#     data = np.random.rand(10, 2)
#     plt.cla()
#     plt.scatter(data[:, 0], data[:, 1])
#     plt.pause(0.1)

for i in range(20):
    data = np.random.rand(20, 20)
    plt.cla()
    plt.plot(data.T)
    plt.pause(1e-10)

# ax = Axes3D(fig)
# for i in range(20):
#     data = np.random.rand(100, 3)
#     plt.cla()
#     ax.scatter(data[:, 0], data[:, 1], data[:, 2])
#     plt.show()
#     plt.pause(1e-5)

plt.ioff()
plt.show()
