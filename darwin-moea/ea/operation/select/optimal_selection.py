import numpy as np
from darwin.ea.base.select import SelectBace


class OptimalSelection(SelectBace):
    def __init__(self, k=2):
        self.k = k
        super(OptimalSelection, self).__init__()

    def do(self, fitness, pop_size):

        indx = np.lexsort(np.rot90(fitness))
        return indx[:pop_size]
