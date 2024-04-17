import numpy as np
from darwin.ea.base.indicator import IndicatorBase


class HV(IndicatorBase):
    def __init__(self, problem):
        super().__init__(problem)

    def evaluate(self, pf, obj_var):
        m = obj_var.shape[1]
        fmin = np.min(np.vstack((np.min(obj_var, axis=0), np.zeros((1,m)))), axis=0)
        fmax = np.max(pf, axis=0) * 1.1
        obj_var = (obj_var - fmin) / (fmax - fmin)
        obj_var = obj_var[np.any(obj_var <= 1, axis=1)]
        ref_point = np.ones((1, m))

        if m == 0:
            return 0
        if m == 2:
            obj_var = ref_point - obj_var
            inx = np.argsort(obj_var[:, 0])
            obj_var = obj_var[inx]
            obj_var[1:, 0] -= obj_var[:-1, 0]
            score = np.sum(np.prod(inx, axis=1))
            return score
        else:
            n_sample = 1000000
            f_max = ref_point
            f_min = np.min(obj_var, axis=0)
            samples = np.random.uniform(f_min, f_max, (n_sample, m))
            for i in obj_var:
                flag = np.full(len(samples), True)
                is_in = i <= samples
                flag = flag & np.all(is_in, axis=1)
                samples = samples[~flag]
            score = np.prod(f_max - f_min) * (1 - len(samples)/n_sample)
            return score


if __name__ == "__main__":
    from darwin.algorithm.nsga2 import NSGA2
    alg = NSGA2()
    res = []
    for i in range(200):
        alg.next()
        F = alg.pop.F
        front, MaxFNo = alg.non_dominated_sort.do(F)
        hv = HV(alg.problem)
        score = hv.evaluate(alg.problem.pareto_front(), F[np.where(front == 0)])
        print(score)
        res.append(score)
