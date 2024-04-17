import numpy as np
from .problem.DTLZ import DTLZ1, DTLZ2, DTLZ3, DTLZ4


class Settings:
    def __init__(self,
                 population_num=100,
                 objectives_num=3,
                 dec_num=30,
                 evaluation=100,
                 problem=DTLZ1,
                 encoding="real",
                 visualization=True):

        self.population_num = population_num

        self.objectives_num = objectives_num

        self.dec_num = dec_num

        self.evaluation = evaluation

        self.problem = problem

        self.encoding = encoding

        self.visualization = visualization
