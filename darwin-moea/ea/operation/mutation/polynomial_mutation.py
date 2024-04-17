import copy
import numpy as np
from darwin.ea.base.mutation import MutationBase
from darwin.ea.operation.repair.out_of_bounds_repair import out_of_bounds_repair


class PolynomialMutation(MutationBase):
    def __init__(self, prob=1.0, eta=20.0):
        self.prob = prob
        self.eta = eta
        super().__init__()

    def do(self, bounds, genes):

        offspring = copy.deepcopy(genes)
        N, D = offspring.shape

        lower = np.tile(bounds['xl'], (N, 1))
        upper = np.tile(bounds['xu'], (N, 1))
        delta1 = (offspring - lower) / (upper - lower)
        delta2 = (upper - offspring) / (upper - lower)

        rand = np.random.random((N, D))
        delta = np.zeros((N, D))
        # 变异概率
        mutation = np.random.random((N, D)) < (self.prob / D)

        # rand <= 0.5
        delta[rand <= 0.5] = 2 * rand[rand <= 0.5] + (1 - 2 * rand[rand <= 0.5]) * np.power(1 - delta1[rand <= 0.5], self.eta + 1.0)
        delta[rand <= 0.5] = np.power(delta[rand <= 0.5], 1.0 / (self.eta + 1)) - 1.0

        # rand > 0.5
        delta[rand > 0.5] = 2.0 * (1.0 - rand[rand > 0.5]) + 2.0 * (rand[rand > 0.5] - 0.5) * np.power(1 - delta2[rand > 0.5], self.eta + 1.0)
        delta[rand > 0.5] = 1.0 - np.power(delta[rand > 0.5], 1.0 / (self.eta + 1))

        offspring[mutation] = offspring[mutation] + delta[mutation] * (upper[mutation] - lower[mutation])

        offspring = out_of_bounds_repair(bounds, offspring)
        return offspring


if __name__ == "__main__":
    from darwin.problem.DTLZ import DTLZ1
    problem1 = DTLZ1(n_var=30, n_obj=3)
    genes = np.random.random((2, 30))
    pm = PolynomialMutation()
    bounds = problem1.bounds
    offs = pm.do(bounds=bounds, genes=genes)

"""
site = np.random.random((N, D)) < self.prob / D
        mu = np.random.random((N, D))

        temp = site & (mu <= 0.5)
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * ((2. * mu[temp] + (1 - 2 * mu[temp]) * \
        (1 - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])) ** (self.eta + 1)) ** (1 / (self.eta + 1)) - 1)

        temp = site & (mu > 0.5)
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (
                    1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * \
            (1 - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])) ** (self.eta + 1)) ** (1 / (self.eta + 1)))
        


"""