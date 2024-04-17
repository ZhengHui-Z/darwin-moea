from darwin.algorithm.nsga2 import NSGA2
from darwin.algorithm.moead import MOEAD


alg = MOEAD()
alg.evolve()


alg2 = NSGA2()
alg2.evolve()


