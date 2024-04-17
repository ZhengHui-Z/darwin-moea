from darwin.algorithm.nsga2_bk import NSGA2
from darwin.settings import Settings



settings = Settings()
settings.visualization = True
settings.evaluation = 80

alg = NSGA2(settings)

alg.run()


