from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize

problem = get_problem("dtlz1", n_var=30, n_obj=3)

algorithm = MOEAD(
    get_reference_directions("das-dennis", 3, n_partitions=12),
    n_neighbors=15,
    decomposition="pbi",
    prob_neighbor_mating=0.7
)

res = minimize(problem, algorithm, termination=('n_gen', 500))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(5, 5))
fig = Axes3D(fig)
data = res.F
fig.scatter(data[:, 0], data[:, 1], data[:, 2])
plt.pause(1e-5)
