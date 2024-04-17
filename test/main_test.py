import geatpy as ea # Import geatpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class moea_NSGA2_templet(ea.MoeaAlgorithm):
    """
moea_NSGA2_templet : class - 多目标进化NSGA-II算法模板

算法描述:
    采用NSGA-II进行多目标优化，算法详见参考文献[1]。

模板使用注意:
    本模板调用的目标函数形如：aimFunc(pop),
    其中pop为Population类的对象，代表一个种群，
    pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵，
    该函数根据该Phen计算得到种群所有个体的目标函数值组成的矩阵，并将其赋值给pop对象的ObjV属性。
    若有约束条件，则在计算违反约束程度矩阵CV后赋值给pop对象的CV属性（详见Geatpy数据结构）。
    该函数不返回任何的返回值，求得的目标函数值保存在种群对象的ObjV属性中，
                          违反约束程度矩阵保存在种群对象的CV属性中。
    例如：population为一个种群对象，则调用aimFunc(population)即可完成目标函数值的计算，
         此时可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。
    若不符合上述规范，则请修改算法模板或自定义新算法模板。

参考文献:
    [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective
    genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary
    Computation, 2002, 6(2):0-197.

    """

    def __init__(self, problem, population):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if str(type(population)) != "<class 'Population.Population'>":
            raise RuntimeError('传入的种群对象必须为Population类型')
        self.name = 'NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=1)  # 生成二进制变异算子对象
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
            self.mutOper = ea.Mutpolyn(Pm=1, DisI=20)  # 生成多项式变异算子对象
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
        self.child = []
        self.jilu = []

    def reinsertion(self, population, offspring, NUM):

        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
            注：这里对原版NSGA-II进行等价的修改：先按帕累托分级和拥挤距离来计算出种群个体的适应度，
            然后调用dup选择算子(详见help(ea.dup))来根据适应度从大到小的顺序选择出个体保留到下一代。
            这跟原版NSGA-II的选择方法所得的结果是完全一样的。
        """

        # 父子两代合并
        population = population + offspring
        # 选择个体保留到下一代
        [levels, criLevel] = self.ndSort(self.problem.maxormins * population.ObjV, NUM, None,
                                         population.CV)  # 对NUM个个体进行非支配分层
        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        return population[chooseFlag]

    def run(self):
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        if population.Chrom is None:
            population.initChrom()  # 初始化种群染色体矩阵（内含解码，详见Population类的源码）
        else:
            population.Phen = population.decoding()  # 染色体解码
        self.problem.aimFunc(population)  # 计算种群的目标函数值
        self.evalsNum = population.sizes  # 记录评价次数
        [levels, criLevel] = self.ndSort(self.problem.maxormins * population.ObjV, NIND, None,
                                         population.CV)  # 对NIND个个体进行非支配分层
        population.FitnV[:, 0] = 1 / levels  # 直接根据levels来计算初代个体的适应度
        # ===========================开始进化============================
        while self.terminated(population) == False:

            # 选择个体参与进化
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 对选出的个体进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            # print(offspring.Field)
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
            offspring.Phen = offspring.decoding()  # 解码
            self.problem.aimFunc(offspring)  # 求进化后个体的目标函数值
            self.child.append(offspring.ObjV)
            self.evalsNum += offspring.sizes  # 更新评价次数
            # 重插入生成新一代种群
            population = self.reinsertion(population, offspring, NIND)

        return self.finishing(population)  # 调用finishing完成后续工作并返回结果


class MyProblem(ea.Problem): # Inherited from Problem class.
    def __init__(self, M): # M is the number of objects.
        name = 'DTLZ1' # Problem's name.
        maxormins = [1] * M # All objects are need to be minimized.

        Dim = 30 # Set the dimension of decision variables.
        varTypes = [0] * Dim # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [0] * Dim # The lower bound of each decision variable.
        ub = [1] * Dim # The upper bound of each decision variable.
        lbin = [1] * Dim # Whether the lower boundary is included.
        ubin = [1] * Dim # Whether the upper boundary is included.
        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def aimFunc(self, pop): # Write the aim function here, pop is an object of Population class.
        Vars = pop.Phen # Get the decision variables
        XM = Vars[:,(self.M-1):]
        g = np.array([100 * (self.Dim - self.M + 1 + np.sum(((XM - 0.5)**2 - np.cos(20 * np.pi * (XM - 0.5))), 1))]).T
        ones_metrix = np.ones((Vars.shape[0], 1))
        pop.ObjV = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_metrix, Vars[:, :self.M-1]]), 1)) * np.hstack([ones_metrix, 1 - Vars[:, range(self.M - 2, -1, -1)]]) * np.tile(1 + g, (1, self.M))
    def calBest(self): # Calculate the theoretic global optimal solution here.
        uniformPoint, ans = ea.crtup(self.M, 10000) # create 10000 uniform points.
        realBestObjV = uniformPoint / 2
        return realBestObjV

"""=========================Instantiate your problem=========================="""
M = 3                     # Set the number of objects.
problem = MyProblem(M)     # Instantiate MyProblem class
"""===============================Population set=============================="""
Encoding = 'RI'            # Encoding type.
NIND = 100                 # Set the number of individuals.
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # Create the field descriptor.
population = ea.Population(Encoding, Field, NIND) # Instantiate Population class(Just instantiate, not initialize the population yet.)
"""================================Algorithm set==============================="""
myAlgorithm = moea_NSGA2_templet(problem, population) # Instantiate a algorithm class.
myAlgorithm.MAXGEN = 100 # Set the max times of iteration.
"""===============================Start evolution=============================="""
import time
t1 = time.time()
NDSet = myAlgorithm.run() # Run the algorithm templet.
print("时间花费：", time.time() - t1)
"""=============================Analyze the result============================="""
PF = problem.calBest() # Get the global pareto front.
GD = ea.indicator.GD(NDSet.ObjV, PF) # Calculate GD
IGD = ea.indicator.IGD(NDSet.ObjV, PF) # Calculate IGD
HV = ea.indicator.HV(NDSet.ObjV, PF) # Calculate HV
# Space = ea.indicator.spacing(NDSet.ObjV) # Calculate Space
print('The number of non-dominated result: %s'%(NDSet.sizes))
print('GD: ', GD)
print('IGD: ', IGD)
print('HV: ', HV)
# print('Space: ', Space)
if PF is not None:
    metricName = [['IGD'], ['GD']]
    [NDSet_trace, Metrics] = ea.indicator.moea_tracking(myAlgorithm.pop_trace, PF, metricName, problem.maxormins)
    # 绘制指标追踪分析图
    ea.trcplot(Metrics, labels = metricName, titles = metricName)

"""=============================迭代图============================="""

fig = plt.figure()
fig = Axes3D(fig)
for i in myAlgorithm.pop_trace:
    plt.cla()
    fig.scatter(i.ObjV[:, 0], i.ObjV[:, 1], i.ObjV[:, 2])
    plt.pause(0.0001)

"""=============================Analyze the result============================="""

