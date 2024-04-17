class Individual:
    __name__ = 'individual'

    def __init__(self, X=None, F=None, cv=0, G=None, Fitness=0, feasible=None, **kwargs) -> None:
        self.X = X
        self.F = F
        self.CV = cv
        self.G = G
        self.Fitness = Fitness
        self.feasible = feasible
        self.data = kwargs

    def set(self, attr, var):
        self.__dict__[attr] = var


if __name__ == "__main__":
    ind = Individual(X=0.3)
