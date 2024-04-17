import numpy as np
from darwin.ea.base.select import SelectBace


class TournamentSelection(SelectBace):
    def __init__(self, k=5):
        self.k = k
        super(TournamentSelection, self).__init__()

    def do(self, fitness, pop_size):
        fitness = fitness.reshape(-1)
        indx = np.random.randint(0, len(fitness), (self.k, pop_size))
        best = np.argmin(fitness[indx], 0)
        return indx[best, range(pop_size)]
