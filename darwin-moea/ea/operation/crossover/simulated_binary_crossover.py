import copy
import numpy as np
from darwin.ea.base.crossover import CrossoverBase
from darwin.ea.operation.repair.out_of_bounds_repair import out_of_bounds_repair


class SimulatedBinaryCrossover(CrossoverBase):
    def __init__(self, eta: float = 20, prob_per_variable: float = 1, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)
        self.eta = eta
        self.__EPS = 1.0e-14
        self.prob_per_variable = prob_per_variable

    def do(self, bounds, genes):
        """
        参考：https://blog.csdn.net/u013555719/article/details/97395038
        :param bounds:
        :param genes:
        :return:
        """
        genes = copy.deepcopy(genes)
        if genes.shape[0] % 2 != 0:
            genes = np.delete(genes, -1, 0)
        parent1, parent2 = np.array_split(genes, 2, 0)

        N, D = parent1.shape
        rand = np.random.random((N, D))
        beta = np.zeros((N, D))

        beta[rand <= 0.5] = np.power((2 * rand[rand <= 0.5]), (1 / (self.eta + 1)))
        beta[rand > 0.5] = np.power(1 / (2 - 2 * rand[rand > 0.5]), 1 / (self.eta + 1))

        # 交换个体某一个维度的值，增加个体多样性
        beta = beta * (-1) ** np.random.randint(0, 2, size=(N, D))

        # 根据交叉概率判断是否进行交叉
        beta[np.random.random((N, D)) < 0.5] = 1
        beta[np.random.random((N, D)) > self.prob_per_variable] = 1

        offspring = np.vstack(
            (
                (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2,
                (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
            )
        )
        offspring = out_of_bounds_repair(bounds, offspring)

        return offspring


if __name__ == "__main__":
    from darwin.problem.DTLZ import DTLZ1
    problem = DTLZ1(n_var=30, n_obj=3)
    genes = np.random.random((2, 30))
    sbx = SimulatedBinaryCrossover()
    bounds = problem.bounds
    offs = sbx.do(bounds=bounds, genes=genes)
