import abc
from darwin.ea.operation.sampling import random_sampling
from darwin.ea.operation.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from darwin.ea.operation.mutation.polynomial_mutation import PolynomialMutation


class AlgorithmBase(metaclass=abc.ABCMeta):
    def __init__(self, settings, name="", crossover=None, mutation=None, **kwargs):
        self.settings = settings
        self.current_evolution = 1
        self.name = name
        if crossover is None:
            self.crossover = SimulatedBinaryCrossover(prob=1.0, eta=20)
        else:
            self.crossover = crossover

        if mutation is None:
            self.mutation = PolynomialMutation(prob=1.0, eta=20)
        else:
            self.mutation = mutation

        if self.settings.visualization:
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            self.fig = plt.figure(figsize=(5, 5))
            if self.settings.objectives_num == 3:
                from mpl_toolkits.mplot3d import Axes3D
                self.fig = Axes3D(self.fig)
                self.fig.view_init(elev=15, azim=10)

    # def init(self):
    #     pass

    @abc.abstractmethod
    def next(self):
        pass

    def evolve(self):
        while self.terminated():
            self.next()

    def terminated(self):
        if self.current_evolution > self.settings.evaluation:
            # self.observer()
            return False
        else:
            # print(self.current_evolution)
            self.observer()
            self.current_evolution += 1
            return True

    def observer(self):
        if self.settings.visualization:
            self.draw(self.pop.F, self.current_evolution)

    def draw(self, data, info=None):
        self.plt.cla()
        self.plt.title("{0}--evaluation: {1}".format(self.name, info))
        obj_num = self.settings.objectives_num
        if obj_num == 2:
            self.plt.scatter(data[:, 0], data[:, 1])
        elif obj_num == 3:
            self.fig.scatter(data[:, 0], data[:, 1], data[:, 2])
        else:
            self.plt.plot(data.T)
        self.plt.pause(1e-10)

