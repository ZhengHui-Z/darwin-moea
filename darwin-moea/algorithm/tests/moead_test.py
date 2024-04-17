from darwin.algorithm.moead_bk import MOEAD
from darwin.ea.population import Population
from darwin.settings import Settings
from darwin.problem.DTLZ import DTLZ1
import time
# https://www.cnblogs.com/timssd/p/4806010.html


settings = Settings()
settings.visualization = True
settings.evaluation = 80

alg = MOEAD(settings)

alg.run()

