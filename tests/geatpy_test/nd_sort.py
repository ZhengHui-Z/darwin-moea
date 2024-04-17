import numpy as np
import geatpy as ea
from darwin.problem.DTLZ import DTLZ1
from darwin.ea.operation.dominated.fast_non_dominated_sort import FastNonDominatedSort

x = np.random.random((1000, 30))

problem = DTLZ1(n_obj=3, n_var=30)

y = problem.calculate(x)

[levels, criLevel] = ea.ndsortESS(y, 1000, None, np.zeros((1000, 1)))


non_dominated_sort = FastNonDominatedSort()

[front, MaxFNo] = non_dominated_sort.do(y)

front = front.astype(levels.dtype) + 1.0
print(np.where(front != levels))
