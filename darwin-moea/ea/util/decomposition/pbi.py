from darwin.ea.base.decomposition import DecompositionBase
from darwin.ea.util.decomposition import tools


class PBI(DecompositionBase):

    def __init__(self, theta=5):
        super(PBI, self).__init__()
        self.theta = theta

    def do(self, F, weights,  utopian_point, **kwargs):
        d1, d2 = tools.calc_distance_to_weights(F=F, weights=weights, utopian_point=utopian_point)
        return d1 + self.theta * d2
