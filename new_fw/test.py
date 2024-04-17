import numpy as np
from darwin.ea.individual import Individual

ndarray = np.ndarray

class Population(ndarray):
    def __new__(subtype, individual_n=0, X=None, F=None, CV=None, individual=Individual(), **kwargs):
        # parameters = {'X': X, 'F': F, 'CV': CV}
        # individual_n = len(X)
        self = ndarray.__new__(subtype, shape=individual_n, dtype=individual.__class__, order='C')
        self[:] = [Individual() for i in range(individual_n)]
        return self

    def init(self, problem, genes):
        character = problem.calculate(genes)
        for i in range(len(genes)):
            self[i].X = genes[i]
            self[i].F = character[i]

    def merge(self, other):
        obj = np.concatenate([self, other]).view(Population)
        return obj

    def set(self):
        pass


    def new(self, genes):
        return Population(genes=genes)

    def pprint(self, test):
        print(test)

    def __add__(self, other):
        obj = np.concatenate([self, other]).view(Population)
        return obj

    def __iadd__(self, other):
        obj = np.concatenate([self, other]).view(Population)
        return obj

    def __getattribute__(self, attr):
        if attr in ['X', 'F', 'CV']:
            if self[0].__dict__[attr] is None:
                raise Exception("{0} is None".format(attr))
            else:
                res = [self[i].__dict__[attr] for i in range(self.shape[0])]
                return res
        try:
            return ndarray.__getattribute__(self, attr)
        except AttributeError:
            pass

if __name__ == "__main__":
    pop = Population(100)
    pop.pprint("Hello")