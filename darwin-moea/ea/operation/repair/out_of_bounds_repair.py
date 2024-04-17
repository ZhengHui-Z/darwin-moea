import numpy as np


def out_of_bounds_repair(bounds, genes):

    only_1d = (genes.ndim == 1)

    if bounds['xl'] is not None:
        xl = np.repeat(bounds['xl'], genes.shape[0], axis=0)
        genes[genes < xl] = xl[genes < xl]

    if bounds['xu'] is not None:
        xu = np.repeat(bounds['xu'], genes.shape[0], axis=0)
        genes[genes > xu] = xu[genes > xu]

    if only_1d:
        return genes[0, :]
    else:
        return genes

