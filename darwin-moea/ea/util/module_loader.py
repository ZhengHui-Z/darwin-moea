import importlib


def loader(module_name):
    return importlib.import_module(module_name)

