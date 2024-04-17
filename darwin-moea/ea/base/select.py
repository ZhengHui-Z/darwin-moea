from abc import ABCMeta, abstractmethod


class SelectBace(object):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def do(self, *args):
        pass
