import numpy as np


def dominate(gene1: np.ndarray, gene2: np.ndarray):
    obj_num = gene1.shape[0]
    if np.sum(gene1 <= gene2) == obj_num and np.sum(gene1 < gene2) > 0:
        return True
    else:
        return False