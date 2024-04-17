import numpy as np
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter

# The pareto front of a scaled zdt1 problem
pf = get_problem("zdt1").pareto_front()

# The result found by an algorithm
A = pf[::10] * 1.1

# plot the result
Scatter(legend=True).add(pf, label="Pareto-front").add(A, label="Result").show()


from pymoo.factory import get_performance_indicator

hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
print("hv", hv.calc(A))