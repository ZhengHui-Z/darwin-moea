import numpy as np
import geatpy as ea


class MyProblem(ea.Problem): # Inherited from Problem class.
    def __init__(self, M): # M is the number of objects.
        name = 'DTLZ1' # Problem's name.
        maxormins = [1] * M # All objects are need to be minimized.
        Dim = 30 # Set the dimension of decision variables.
        varTypes = [0] * Dim # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [0] * Dim # The lower bound of each decision variable.
        ub = [1] * Dim # The upper bound of each decision variable.
        lbin = [1] * Dim # Whether the lower boundary is included.
        ubin = [1] * Dim # Whether the upper boundary is included.
        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, Phen): # Write the aim function here, pop is an object of Population class.
        Vars = Phen # Get the decision variables
        XM = Vars[:, (self.M-1):]
        g = np.array([100 * (self.Dim - self.M + 1 + np.sum(((XM - 0.5)**2 - np.cos(20 * np.pi * (XM - 0.5))), 1))]).T
        ones_metrix = np.ones((Vars.shape[0], 1))
        ObjV = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_metrix, Vars[:,:self.M-1]]), 1)) * np.hstack([ones_metrix, 1 - Vars[:, range(self.M - 2, -1, -1)]]) * np.tile(1 + g, (1, self.M))
        return g, ObjV
    def calBest(self): # Calculate the theoretic global optimal solution here.
        uniformPoint, ans = ea.crtup(self.M, 10000) # create 10000 uniform points.
        realBestObjV = uniformPoint / 2
        return realBestObjV


def evaluate(dec_var) -> np.ndarray:
    n = dec_var.shape[1]
    m = 3
    k = n - m + 1
    x_m = dec_var[:, m-1:]
    # g = np.array([100 * (k + np.sum(((XM - 0.5) ** 2 - np.cos(20 * np.pi * (XM - 0.5))), 1))]).T
    g = np.array([100 * (k + np.sum(((x_m - 0.5) ** 2 - np.cos(20 * np.pi * (x_m - 0.5))), 1))]).T
    ones_matrix = np.ones((dec_var.shape[0], 1))
    f = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_matrix, dec_var[:, :m - 1]]), 1)) * np.hstack(
        [ones_matrix, 1 - dec_var[:, range(m - 2, -1, -1)]]) * np.tile(1 + g, (1, m))

    return f


def aimFunc(Phen):
    M = 3
    Dim = 30
    Vars = Phen
    XM = Vars[:, (M-1):]
    g = np.array([100 * (Dim - M + 1 + np.sum(((XM - 0.5)**2 - np.cos(20 * np.pi * (XM - 0.5))), 1))]).T
    ones_metrix = np.ones((Vars.shape[0], 1))
    f = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_metrix, Vars[:,:M-1]]), 1)) * np.hstack([ones_metrix, 1 - Vars[:, range(M - 2, -1, -1)]]) * np.tile(1 + g, (1, M))
    return f


if __name__ == "__main__":
    var = np.array([[0.19269069, 0.58438063, 0.4420659, 0.24919518, 0.29218244, 0.1123021, 0.40824578, 0.79214261,
                     0.70404882, 0.05390683, 0.76168699, 0.9230519, 0.00883714, 0.01774132, 0.88141423, 0.73107514,
                     0.28863875, 0.57159938, 0.62214353, 0.64493735, 0.40808947, 0.00959348, 0.28168211, 0.00450627,
                     0.4286321, 0.3230117, 0.42624985, 0.41997293, 0.86336331, 0.82665024]])
    f1 = aimFunc(var)
    f2 = evaluate(var)
