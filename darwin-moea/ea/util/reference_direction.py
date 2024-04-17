import numpy as np
from scipy import special


def get_number_of_uniform_points(n_partitions, n_dim):
    """
    Returns the number of uniform points that can be created uniformly.
    """
    return int(special.binom(n_dim + n_partitions - 1, n_partitions))


def get_partition_closest_to_points(n_points, n_dim):
    """
    Returns the corresponding partition number which create the desired number of points
    or less!
    """

    if n_dim == 1:
        return 0

    n_partitions = 1
    _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    while _n_points <= n_points:
        n_partitions += 1
        _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    return n_partitions - 1


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


class UniformReferenceDirection:
    def __init__(self, n_dim, n_points=None, n_partitions=None, **kwargs):
        self.n_dim = n_dim
        if n_points is not None:
            n_partitions = get_partition_closest_to_points(n_points, n_dim)

            self.n_partitions = n_partitions
        elif n_partitions is not None:
            self.n_partitions = n_partitions

    def do(self):
        return das_dennis(self.n_partitions, self.n_dim)


if __name__ == "__main__":
    a = UniformReferenceDirection(n_dim=3, n_points=100).do()


