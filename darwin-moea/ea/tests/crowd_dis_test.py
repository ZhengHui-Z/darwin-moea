import numpy as np
from darwin.ea.operation import nd_sort, crowd_dis

obj_var = np.load("obj_var.npy")

f, max_f = nd_sort(obj_var)
dis = crowd_dis(obj_var)

