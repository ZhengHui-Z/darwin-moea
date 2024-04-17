import numpy as np
from darwin.settings import Settings
from darwin.ea.base.algorithm import AlgorithmBase
from darwin.ea.population import Population
from darwin.ea.operation.dominated.fast_non_dominated_sort import FastNonDominatedSort
from darwin.ea.operation.sampling import random_sampling
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()


class NSGA2(AlgorithmBase):

    def __init__(self, settings=Settings(), **kwargs):
        self.settings = settings
        self.n_var = self.settings.dec_num
        self.n_obj = self.settings.objectives_num
        self.problem = settings.problem(self.n_var, self.n_obj)
        self.pop_size = self.settings.population_num
        self.non_dominated_sort = FastNonDominatedSort()
        super().__init__(settings, name="NSGA2", crossover=None, mutation=None, **kwargs)

        self.pop = Population(random_sampling.float_random_sampling(self.problem.bounds, self.pop_size, self.n_var))
        self.pop.refresh(self.problem)

    def select(self):
        front, MaxFNo =self.non_dominated_sort.do(self.pop.F)
        dis = crowd_dis(self.pop.F, front)
        fit = np.lexsort((-dis, front))
        self.pop = self.pop[fit[:100]]

    def next(self):
        X = matlab.double(self.pop.X.tolist())
        off_x = np.array(eng.GA(X))
        # off_x = self.crossover.do(self.problem.bounds, genes=self.pop.X)
        # off_x = self.mutation.do(self.problem.bounds, genes=off_x)
        off_pop = Population(off_x)
        off_pop.refresh(self.problem)

        self.pop += off_pop
        self.select()


def crowd_dis(obj_var, front_no):
    N, M = obj_var.shape
    crowd_dis = np.zeros(N)
    MaxFNo = np.max(front_no)
    for f in range(MaxFNo):
        front = np.where(front_no == f)[0]
        f_obj = obj_var[front]
        Fmax = np.max(f_obj, axis=0)
        Fmin = np.min(f_obj, axis=0)
        for i in range(M):
            index = np.argsort(f_obj[:, i], axis=0)
            crowd_dis[front[index[0]]] = 1e16
            crowd_dis[front[index[-1]]] = 1e16
            for j in range(1, f_obj.shape[0] - 1):
                crowd_dis[front[index[j]]] += (f_obj[index[j + 1]][i] - f_obj[index[j - 1]][i]) / (
                            Fmax[i] - Fmin[i])
    return crowd_dis


if __name__ == "__main__":
    from darwin.settings import Settings
    from darwin.problem.DTLZ import DTLZ1

    st = Settings(problem=DTLZ1, objectives_num=3, dec_num=30)

    alg = NSGA2(settings=st)
    alg.evolve()
