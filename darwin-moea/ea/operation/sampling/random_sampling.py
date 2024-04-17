import numpy as np
from darwin.ea.util.normalization import denormalize


def float_random_sampling(bounds, n_samples, n_var):
    val = np.random.random((n_samples, n_var))
    return denormalize(val, bounds['xl'], bounds['xu'])


def binary_random_sampling(n_samples, n_var):
    val = np.random.random((n_samples, n_var))
    return (val < 0.5).astype(np.bool)


