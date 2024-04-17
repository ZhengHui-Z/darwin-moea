import numpy as np
from scipy.spatial.distance import cdist
from darwin.ea.base.indicator import IndicatorBase


class GD(IndicatorBase):
    def __init__(self, problem):
        super().__init__(problem)

    def evaluate(self, pf, obj_var):
        distance = np.min(cdist(obj_var, pf), axis=1)
        score = np.linalg.norm(distance, ord=2) / distance.size
        return score
