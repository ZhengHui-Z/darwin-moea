from abc import abstractmethod


class DecompositionBase(object):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def do(self, **kwargs):
        pass
