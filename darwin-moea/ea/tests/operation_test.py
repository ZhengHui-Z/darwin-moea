import numpy as np
from darwin.settings import Settings
from darwin.ea.population import Population
from darwin.problem.DTLZ import DTLZ1
from darwin.ea.operation import cross, mutation

settings = Settings()
settings.problem = DTLZ1
Settings.objectives_num = 5
Settings.dec_num = 10
Settings.visualization = True


pop = Population()

new_gen = cross(pop.dec_var[:2, :], offspring_num=1)

new_gen = mutation(new_gen)


