import numpy as np
from scipy.spatial.distance import cdist
from darwin.settings import Settings
from darwin.ea.base.algorithm import AlgorithmBase
from darwin.ea.population import Population
from darwin.ea.util.reference_direction import UniformReferenceDirection
from darwin.ea.util.decomposition.pbi import PBI
from darwin.ea.operation.sampling import random_sampling


class MOEAD(AlgorithmBase):
    def __init__(self, settings=None, n_neighbors=15, crossover=None, mutation=None, prob_neighbor_mating=1, **kwargs):
        if settings is None:
            settings = Settings()
        super().__init__(settings=settings, name="MOEAD", crossover=crossover, mutation=mutation)

        self.prob_neighbor_mating = prob_neighbor_mating
        self.n_var = self.settings.dec_num

        self.n_obj = self.settings.objectives_num

        self.problem = settings.problem(self.n_var, self.n_obj)

        self.ref_dirs = UniformReferenceDirection(n_dim=self.n_obj, n_points=self.settings.population_num).do()

        self.pop_size = self.ref_dirs.shape[0]
        self.settings.population_num = self.ref_dirs.shape[0]

        self.pop = Population(random_sampling.float_random_sampling(self.problem.bounds, self.pop_size, self.n_var))
        self.pop.refresh(self.problem)
        self.ideal_point = np.min(self.pop.F, axis=0)
        self.n_neighbors = n_neighbors
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]
        self.decomposition = PBI()

    def next(self):
        for i in np.random.permutation(self.pop_size):
            N = self.neighbors[i, :]
            if np.random.random() < self.prob_neighbor_mating:
                parents = N[np.random.permutation(self.n_neighbors)][:2]
            else:
                parents = np.random.permutation(self.pop.population_num)[:2]

            off_x = self.crossover.do(self.problem.bounds, genes=self.pop.X[parents])
            off_x = self.mutation.do(self.problem.bounds, genes=off_x)
            off_x = off_x[np.random.randint(0, len(off_x)), None]
            off_pop = Population(off_x)
            off_pop.refresh(self.problem)

            off_f = off_pop.F
            self.ideal_point = np.min(np.vstack([self.ideal_point, off_f]), axis=0)

            FV = self.decomposition.do(self.pop.F[N, :], weights=self.ref_dirs[N, :], utopian_point=self.ideal_point)
            FV_off = self.decomposition.do(off_f, weights=self.ref_dirs[N, :], utopian_point=self.ideal_point)
            I = np.where(FV_off < FV)[0]
            self.pop[N[I]] = off_pop


if __name__ == "__main__":
    alg = MOEAD()
    alg.evolve()

