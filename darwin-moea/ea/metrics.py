import numpy as np
from scipy.spatial.distance import cdist


def GD(pop_obj, pf):
    distance = np.min(cdist(pop_obj, pf), axis=1)
    score = np.linalg.norm(distance, ord=2) / distance.size
    return score


def IGD(pop_obj, pf):
    distance = np.min(cdist(pop_obj, pf), axis=1)
    score = np.mean(distance, ord=2)
    return score


def HV():
    pass
