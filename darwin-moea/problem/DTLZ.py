import numpy as np
from darwin.ea.base.problem import ProblemBase
from darwin.ea.util.reference_direction import UniformReferenceDirection


def generic_sphere(ref_dirs):
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))


class DTLZ1(ProblemBase):
    def __init__(self, n_var, n_obj):
        xu = np.ones(shape=(1, n_var))
        xl = np.zeros(shape=(1, n_var))
        bounds = {'xu': xu, 'xl': xl}
        super(DTLZ1, self).__init__(n_var=n_var, n_obj=n_obj, name="DTLZ1", n_constr=0, bounds=bounds)

    def calculate(self, dec_var) -> np.ndarray:
        n = self.n_var
        m = self.n_obj
        k = n - m + 1
        x_m = dec_var[:, -k:]
        g = np.array([100 * (k + np.sum(((x_m - 0.5) ** 2 - np.cos(20 * np.pi * (x_m - 0.5))), 1))]).T
        ones_matrix = np.ones((dec_var.shape[0], 1))
        f = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_matrix, dec_var[:, :m - 1]]), 1)) * np.hstack(
            [ones_matrix, 1 - dec_var[:, range(m - 2, -1, -1)]]) * np.tile(1 + g, (1, m))

        return f

    def pareto_front(self, n_points=10000):

        point = UniformReferenceDirection(n_dim=self.n_obj, n_points=n_points).do()
        point = point * 0.5
        return point


class DTLZ2(ProblemBase):
    def __init__(self, n_var, n_obj):
        xu = np.ones(shape=(1, n_var))
        xl = np.zeros(shape=(1, n_var))
        bounds = {'xu': xu, 'xl': xl}
        super(DTLZ2, self).__init__(n_var=n_var, n_obj=n_obj, n_constr=0, bounds=bounds)

    def calculate(self, dec_var) -> np.ndarray:
        M = self.n_obj
        n = self.n_var
        k = n - M + 1
        X_M = dec_var[:, -k:]
        g = np.array([np.sum((X_M - 0.5) ** 2, 1)]).T
        ones_metrix = np.ones((dec_var.shape[0], 1))
        f = np.fliplr(np.cumprod(np.hstack([ones_metrix, np.cos(dec_var[:, :M - 1] * np.pi * 0.5)]), 1)) * np.hstack(
            [ones_metrix, np.sin(dec_var[:, range(M - 2, -1, -1)] * np.pi * 0.5)]) * np.tile(1 + g, (1, M))

        return f

    def pareto_front(self, n_points=10000):
        point = UniformReferenceDirection(n_dim=self.n_obj, n_points=n_points).do()
        point = generic_sphere(point)
        return point


class DTLZ3(ProblemBase):

    def __init__(self, n_var, n_obj):
        xu = np.ones(shape=(1, n_var))
        xl = np.zeros(shape=(1, n_var))
        bounds = {'xu': xu, 'xl': xl}
        super(DTLZ3, self).__init__(n_var=n_var, n_obj=n_obj, n_constr=0, bounds=bounds)

    def calculate(self, dec_var) -> np.ndarray:
        M = self.n_obj
        n = self.n_var
        k = n - M + 1
        X_M = dec_var[:, -k:]
        g = np.array([100 * (k + np.sum(((X_M - 0.5) ** 2 - np.cos(20 * np.pi * (X_M - 0.5))), 1))]).T
        ones_metrix = np.ones((dec_var.shape[0], 1))
        f = np.fliplr(np.cumprod(np.hstack([ones_metrix, np.cos(dec_var[:, :M - 1] * np.pi * 0.5)]), 1)) * np.hstack(
            [ones_metrix, np.sin(dec_var[:, range(M - 2, -1, -1)] * np.pi * 0.5)]) * np.tile(1 + g, (1, M))

        return f

    def pareto_front(self, n_points=10000):
        point = UniformReferenceDirection(n_dim=self.n_obj, n_points=n_points).do()
        point = generic_sphere(point)
        return point


class DTLZ4(ProblemBase):

    def __init__(self, n_var, n_obj):
        xu = np.ones(shape=(1, n_var))
        xl = np.zeros(shape=(1, n_var))
        bounds = {'xu': xu, 'xl': xl}
        super(DTLZ4, self).__init__(n_var=n_var, n_obj=n_obj, n_constr=0, bounds=bounds)

    def calculate(self, dec_var) -> np.ndarray:
        M = self.n_obj
        n = self.n_var
        k = n - M + 1
        X_M = dec_var[:, -k:]
        g = np.array([np.sum((X_M - 0.5) ** 2, 1)]).T
        ones_metrix = np.ones((dec_var.shape[0], 1))
        f = np.fliplr(
            np.cumprod(np.hstack([ones_metrix, np.cos(dec_var[:, :M - 1] ** 100 * np.pi * 0.5)]), 1)) * np.hstack(
            [ones_metrix, np.sin(dec_var[:, range(M - 2, -1, -1)] ** 100 * np.pi * 0.5)]) * np.tile(1 + g, (1, M))

        return f

    def pareto_front(self, n_points=10000):
        point = UniformReferenceDirection(n_dim=self.n_obj, n_points=n_points).do()
        point = generic_sphere(point)
        return point


if __name__ == "__main__":
    x = np.random.random((10, 6))
    problem = DTLZ1(n_var=6, n_obj=3)
    y = problem.calculate(x)
