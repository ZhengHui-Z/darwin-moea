import numpy as np
from darwin.settings import Settings
from darwin.ea.base.algorithm import AlgorithmBase
from darwin.ea.population import Population
from darwin.ea.operation.dominated import ndsortESS, fast_non_dominated_sort
from darwin.ea.operation.sampling import random_sampling
import geatpy as ea


class NSGA2(AlgorithmBase):

    def __init__(self, settings=Settings(), **kwargs):
        self.settings = settings
        self.n_var = self.settings.dec_num
        self.n_obj = self.settings.objectives_num
        self.problem = settings.problem(self.n_var, self.n_obj)
        # self.pf = self.problem.pareto_front()
        self.pop_size = self.settings.population_num
        self.non_dominated_sort = ndsortESS
        self.non_dominated_sort1 = fast_non_dominated_sort.FastNonDominatedSort()
        super().__init__(settings, name="NSGA2", crossovere=None, mutation=None, **kwargs)

        self.pop = Population(random_sampling.float_random_sampling(self.problem.bounds, self.pop_size, self.n_var))
        self.pop.refresh(self.problem)
        self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
        self.mutOper = ea.Mutpolyn(Pm=1, DisI=20)

        [levels, criLevel] = self.non_dominated_sort(self.pop.F, self.pop_size, None, self.pop.CV)
        # [levels, criLevel] = self.non_dominated_sort1.do(self.pop.F)
        dis = ea.crowdis(self.pop.F, levels)
        self.pop.Fitness = np.argsort(np.lexsort((dis, -levels)), kind='mergesort')
        # self.pop.Fitness = 1/levels
        self.track = []
        self.child = []

    def select(self):
        [levels, criLevel] = self.non_dominated_sort(self.pop.F, self.pop_size, None, self.pop.CV.reshape(-1, 1))
        # [levels, MaxFNo] = self.non_dominated_sort1.do(self.pop.F)
        dis = ea.crowdis(self.pop.F, levels)

        self.pop.Fitness = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')
        chooseFlag = ea.selecting('dup', self.pop.Fitness, self.pop_size)
        # front, MaxFNo =self.non_dominated_sort.do(self.pop.F)
        # dis = crowd_dis(self.pop.F, front)
        # fit = np.lexsort((-dis, front))
        self.pop = self.pop[chooseFlag]

    def next(self):
        chooseFlag = ea.selecting('tour', self.pop.Fitness, self.pop_size)
        self.pop = self.pop[chooseFlag]
        self.track.append(np.copy(self.pop.F))
        Field = np.zeros((3, self.n_var))
        Field[1, :] += 1.0

        off_x = self.recOper.do(self.pop.X)  # 重组
        off_x = self.mutOper.do('RI', off_x, Field)  # 变异
        # off_x = self.mutation.do(self.problem.bounds, genes=self.pop.X)
        # off_x = self.mutation.do(self.problem.bounds, genes=off_x)
        off_pop = Population(off_x)
        off_pop.refresh(self.problem)
        self.child.append(off_pop.F)

        self.pop += off_pop
        self.select()


def crowd_dis(obj_var, front_no):
    N, M = obj_var.shape
    crowd_dis = np.zeros(N)
    MaxFNo = np.max(front_no)
    for f in range(MaxFNo):
        front = np.where(front_no == f)[0]
        f_obj = obj_var[front]
        Fmax = np.max(f_obj, axis=0)
        Fmin = np.min(f_obj, axis=0)
        for i in range(M):
            index = np.argsort(f_obj[:, i], axis=0)
            crowd_dis[front[index[0]]] = 1e16
            crowd_dis[front[index[-1]]] = 1e16
            for j in range(1, f_obj.shape[0] - 1):
                crowd_dis[front[index[j]]] += (f_obj[index[j + 1]][i] - f_obj[index[j - 1]][i]) / (
                            Fmax[i] - Fmin[i])
    return crowd_dis


if __name__ == "__main__":
    import time
    from darwin.ea.indicator.igd import IGD
    from darwin.ea.indicator.gd import GD
    import matplotlib.pyplot as plt

    t1 = time.time()
    alg = NSGA2()

    alg.evolve()
    # F = alg.pop.F;alg.draw(F)
    # gd = GD(alg.problem)
    # igd = IGD(alg.problem)
    # F = alg.pop.F


    print(time.time() - t1)
    # pf = alg.problem.pareto_front()
    # res_gd = []
    # res_igd = []
    # for i in alg.track:
    #     res_gd.append(gd.evaluate(pf=pf, obj_var=i))
    #     res_igd.append(igd.evaluate(pf=pf, obj_var=i))
    # plt.figure(1)
    # plt.plot(range(len(res_igd)), res_gd, label="gd")
    # plt.title("gd")
    # plt.figure(2)
    # plt.plot(range(len(res_igd)), res_igd, label="igd")
    # plt.title("igd")
    # plt.show()
    # ans = alg.pop.F
