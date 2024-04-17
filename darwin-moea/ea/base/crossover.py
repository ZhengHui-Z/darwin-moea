from abc import abstractmethod


class CrossoverBase(object):
    def __init__(self, n_parents, n_offsprings, prob=0.5):
        self.prob = prob
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings

    @abstractmethod
    def do(self, *args):
        pass

