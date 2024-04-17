import numpy as np
from abc import ABCMeta, abstractmethod


class ProblemBase(object, metaclass=ABCMeta):
    def __init__(self,
                 n_var=-1,
                 n_obj=-1,
                 name=None,
                 n_constr=0,
                 bounds=None,
                 type_var=np.double,
                 ):
        self.n_var = n_var
        self.type_var = type_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.bounds = bounds
        self.name = name

    @abstractmethod
    def calculate(self, *args):
        pass

    @abstractmethod
    def pareto_front(self, *args):
        pass
