import numpy as np
from darwin.ea.individual import Individual

ndarray = np.ndarray


class Population(ndarray):
    def __new__(cls, genes=None, individual_n=0, individual=Individual(), **kwargs):
        if genes is None:
            self = ndarray.__new__(cls, shape=individual_n, dtype=individual.__class__, order='C')
        else:
            individual_n = len(genes)
            self = ndarray.__new__(cls, shape=individual_n, dtype=individual.__class__, order='C')
            self[:] = [Individual(X=genes[i]) for i in range(individual_n)]
        return self

    def init(self, problem, genes):
        character = problem.calculate(genes)
        for i in range(len(genes)):
            self[i].X = genes[i]
            self[i].F = character[i]

    def refresh(self, problem):
        if not isinstance(self.X, np.ndarray):
            X = np.array(self.X)
        else:
            X = self.X
        n = len(self)
        F = problem.calculate(X)
        [self[i].set('F', F[i]) for i in range(n)]

    def merge(self, other):
        obj = np.concatenate([self, other]).view(Population)
        return obj

    def new(self, genes):
        return Population(genes=genes)

    def __add__(self, other):
        obj = np.concatenate([self, other]).view(Population)
        return obj

    def __iadd__(self, other):
        obj = np.concatenate([self, other]).view(Population)
        return obj

    def __getattribute__(self, attr):
        if attr in ['X', 'F', 'CV', 'Fitness', 'G']:
            if self[0].__dict__[attr] is None:
                return None
            else:
                if isinstance(self[0].__dict__[attr], np.ndarray):
                    res = [self[i].__dict__[attr] for i in range(self.shape[0])]
                    return np.array(res)
                else:
                    res = [[self[i].__dict__[attr]] for i in range(self.shape[0])]
                    return np.array(res)
        try:
            return ndarray.__getattribute__(self, attr)
        except AttributeError:
            pass

    def __setattr__(self, name, value):
        if name in ['X', 'F', 'CV', 'Fitness']:
            n = len(self)
            if len(value) != n:
                raise Exception("参数长度{}和种群大小{}不一致".format(len(value), n))
            [self[i].set(name, value[i]) for i in range(n)]
        else:
            super(Population, self).__setattr__(name, value)


if __name__ == "__main__":
    from darwin.problem.DTLZ import DTLZ1
    problem = DTLZ1(n_var=30, n_obj=3)
    var = np.random.random((100, 30))
    pop = Population(var)
    pop.refresh(problem)
    a = pop.CV

