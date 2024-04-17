import abc


class IndicatorBase:
    def __init__(self, problem):
        self.problem = problem

    @abc.abstractmethod
    def evaluate(self, pf, obj_var):
        pass
