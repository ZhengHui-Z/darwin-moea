from abc import ABCMeta, abstractmethod


class EABase(object, metaclass=ABCMeta):

    @abstractmethod
    def crossover(self, **kwargs):
        pass

    @abstractmethod
    def mutation(self, **kwargs):
        pass

    @abstractmethod
    def select(self, **kwargs):
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass


