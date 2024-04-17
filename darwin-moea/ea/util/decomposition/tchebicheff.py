import numpy as np
from darwin.ea.base.decomposition import DecompositionBase


class Tchebicheff(DecompositionBase):

    def __init__(self, **kwargs):
        super(Tchebicheff, self).__init__(**kwargs)

    def do(self, F, weights, utopian_point, **kwargs):
        v = np.abs(F - utopian_point) * weights
        tchebicheff = v.max(axis=1)
        return tchebicheff
