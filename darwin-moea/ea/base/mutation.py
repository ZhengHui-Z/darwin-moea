from abc import abstractmethod


class MutationBase(object):
    def __init__(self):
        pass

    @abstractmethod
    def do(self, **kwargs):
        pass
