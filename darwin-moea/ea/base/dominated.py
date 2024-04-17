from abc import abstractmethod


class DominatedBase(object):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def do(self, F):
        pass
