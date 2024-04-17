import numpy as np
from itertools import combinations
# from ..settings import Settings

"""
    Module function：SBX simulated binary crossover
    Currently only contains sbx algorithm
"""


def cross(gene, dec_constraint, crossover_p=1, crossover_d=20, encode="real", offspring_num=2):
    """
    :param offspring_num:
    :param gene: 待交叉的种群
    :param dec_constraint: 决策变量约束
    :param crossover_p: is the probabilities of doing crossover
    :param crossover_d: is the distribution index of simulated binary crossover
    :param encode: Population gene sequence coding coding method
    :return: 生成的子代种群
    """

    dec_array = np.copy(gene)

    if len(dec_array) % 2 != 0:
        dec_array = np.delete(dec_array, -1, 0)
    parent1, parent2 = np.array_split(dec_array, 2, 0)
    p_num, d_num = parent1.shape
    if encode == "real":
        # Simulated binary crossover
        beta = np.zeros((p_num, d_num))
        mu = np.random.random((p_num, d_num))
        
        beta[mu <= 0.5] = np.power((2 * mu[mu <= 0.5]), (1 / crossover_d + 1))
        beta[mu > 0.5] = np.power(1/(2 - 2 * mu[mu > 0.5]), 1 / (crossover_d + 1))


        # 交换个体某一个维度的值，增加个体多样性
        beta = beta * (-1) ** np.random.randint(0, 2, size=(p_num, d_num))

        # 根据交叉概率判断是否进行交叉
        beta[np.random.random((p_num, d_num)) < 0.5] = 1
        beta[np.random.random((p_num, d_num)) > crossover_p] = 1

        if offspring_num == 2:
            offspring = np.vstack(
                (
                    (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2,
                    (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
                )
            )
            lower = np.tile(dec_constraint(d_num)['lower'], (2 * p_num, 1))
            upper = np.tile(dec_constraint(d_num)['upper'], (2 * p_num, 1))
            offspring = np.minimum(np.maximum(offspring, lower), upper)
        else:
            offspring = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
            lower = np.tile(dec_constraint(d_num)['lower'], (p_num, 1))
            upper = np.tile(dec_constraint(d_num)['upper'], (p_num, 1))
            offspring = np.minimum(np.maximum(offspring, lower), upper)

        return offspring


def repair_out_of_bounds(xl, xu, X):
    xl = np.repeat(xl[None, :], X.shape[0], axis=0)
    xu = np.repeat(xu[None, :], X.shape[0], axis=0)
    X[X < xl] = xl[X < xl]
    X[X > xu] = xu[X > xu]
    return X


def cross2(gene, dec_constraint, crossover_p=1, crossover_d=20, encode="real", offspring_num=2):
    dec_array = np.copy(gene)
    if len(dec_array) % 2 != 0:
        dec_array = np.delete(dec_array, -1, 0)
    parent1, parent2 = np.array_split(dec_array, 2, 0)
    p_num, d_num = parent1.shape
    X = np.array([parent1, parent2])
    eta = 1
    prob_per_variable = 0.5
    xl, xu = dec_constraint(d_num)['lower'], dec_constraint(d_num)['upper']

    # if np.any(X < xl) or np.any(X > xu):
    #    raise Exception("Simulated binary crossover requires all variables to be in bounds!")

    # crossover mask that will be used in the end
    do_crossover = np.full(X[0].shape, True)

    # per variable the probability is then 50%
    do_crossover[np.random.random((p_num, d_num)) > prob_per_variable] = False
    # also if values are too close no mating is done
    do_crossover[np.abs(X[0] - X[1]) <= 1.0e-14] = False

    # assign y1 the smaller and y2 the larger value
    y1 = np.min(X, axis=0)
    y2 = np.max(X, axis=0)

    # random values for each individual
    rand = np.random.random((p_num, d_num))

    def calc_betaq(beta):
        alpha = 2.0 - np.power(beta, -(eta + 1.0))

        mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))

        betaq = np.zeros(mask.shape)
        betaq[mask] = np.power((rand * alpha), (1.0 / (eta + 1.0)))[mask]
        betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)), (1.0 / (eta + 1.0)))[mask_not]

        return betaq

    # difference between all variables
    delta = (y2 - y1)

    # now just be sure not dividing by zero (these cases will be filtered later anyway)
    # delta[np.logical_or(delta < 1.0e-10, np.logical_not(do_crossover))] = 1.0e-10
    delta[delta < 1.0e-10] = 1.0e-10

    beta = 1.0 + (2.0 * (y1 - xl) / delta)
    betaq = calc_betaq(beta)
    c1 = 0.5 * ((y1 + y2) - betaq * delta)

    beta = 1.0 + (2.0 * (xu - y2) / delta)
    betaq = calc_betaq(beta)
    c2 = 0.5 * ((y1 + y2) + betaq * delta)

    # do randomly a swap of variables
    b = np.random.random((p_num, d_num)) <= 0.5
    val = np.copy(c1[b])
    c1[b] = c2[b]
    c2[b] = val

    # take the parents as _template
    c = np.copy(X)

    # copy the positions where the crossover was done
    c[0, do_crossover] = c1[do_crossover]
    c[1, do_crossover] = c2[do_crossover]
    c[0] = repair_out_of_bounds(xl, xu, c[0])
    c[1] = repair_out_of_bounds(xl, xu, c[1])

    return np.vstack((c[0], c[1]))


def mutation(gene, dec_constraint, mutation_p=1, mutation_d=20):
    """
    Polynomial mutation
    :param gene: Population gene
    :param dec_constraint: 决策变量约束
    :param mutation_p: is the expectation of number of bits doing mutation
    :param mutation_d: is the distribution index of polynomial mutation
    :return:
    """
    offspring = np.copy(gene)
    p_num, d_num = offspring.shape

    lower = np.tile(dec_constraint(d_num)['lower'], (p_num, 1))
    upper = np.tile(dec_constraint(d_num)['upper'], (p_num, 1))

    site = np.random.random((p_num, d_num)) < mutation_p / d_num
    mu = np.random.random((p_num, d_num))

    temp = site & (mu <= 0.5)
    offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * \
                      ((2 * mu[temp] + (1 - 2 * mu[temp]) * (
                              1 - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])) ** (
                                mutation_d + 1)) ** (1 / (mutation_d + 1)) - 1)

    temp = site & (mu > 0.5)
    offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * \
                      (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (
                              1 - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])) ** (
                                    mutation_d + 1)) ** (1 / (mutation_d + 1)))

    return offspring


def dominate(gene1: np.ndarray, gene2: np.ndarray):
    obj_num = gene1.shape[0]
    if np.sum(gene1 <= gene2) == obj_num and np.sum(gene1 < gene2) > 0:
        return True
    else:
        return False


def nd_sort(a):
    # a = pop.obj_fun
    N = a.shape[0]
    front = np.full(N, -1, dtype=np.int32)

    index = np.lexsort(np.rot90(a))

    MaxFNo = 0
    for i in index:
        NowFNo = 0
        while True:
            Dominated = False
            for j in np.flipud(np.where(front == NowFNo)[0]):
                if dominate(a[j], a[i]):
                    Dominated = True
                    break
            if Dominated:
                NowFNo += 1
                if NowFNo > MaxFNo:
                    MaxFNo += 1
            else:
                front[i] = NowFNo
                break
    return front, MaxFNo


def crowd_dis(obj_var, front_no):
    N, M = obj_var.shape
    crowd_dis = np.zeros(N)
    MaxFNo = np.max(front_no)
    for f in range(MaxFNo):
        front = np.where(front_no == f)[0]
        f_obj = obj_var[front]
        Fmax = np.max(f_obj, axis=0)
        Fmin = np.min(f_obj, axis=0)
        for i in range(M):
            index = np.argsort(f_obj[:, i], axis=0)
            crowd_dis[front[index[0]]] = np.inf
            crowd_dis[front[index[-1]]] = np.inf
            for j in range(1, f_obj.shape[0] - 1):
                crowd_dis[front[index[j]]] += (f_obj[index[j + 1]][i] - f_obj[index[j - 1]][i]) / (Fmax[i] - Fmin[i])

    return crowd_dis


def combcount(n, r):
    f = lambda n, r: n*f(n-1, r) if n > r else 1
    return int(f(n, n-r) / f(r, 0))


def permcount(n, r):
    f = lambda n, r: n*f(n-1, r) if n > r else 1
    return int(f(n, n-r))


def UniformPoint(N, M):
    H1 = 1
    while combcount(H1 + M, M - 1) <= N: # 从n种情况中一次取出k种的组合的数量
        H1 = H1 + 1
    W = np.array(list(combinations(range(1, H1 + M), M-1))) - np.tile(range(M - 1), (combcount(H1+M-1, M-1), 1)) - 1

    W = (np.hstack((W, np.zeros((W.shape[0], 1)) + H1)) - np.hstack((np.zeros((W.shape[0], 1)), W))) / H1

    if H1 < M:
        H2 = 0
        while combcount(H1+M-1, M-1)+combcount(H2+M, M-1) <= N:
            H2 = H2 + 1

        if H2 > 0:
            W2 = np.array(list(combinations(range(1, H2+M), M-1))) - np.tile(range(0, M-1), (combcount(H2+M-1, M-1), 1)) - 1
            W2 = (np.hstack((W2, np.zeros((W2.shape[0], 1)) + H2)) - np.hstack((np.zeros((W2.shape[0], 1)), W2))) / H2
            W = np.vstack((W, W2/2+1/(2*M)))

    N = W.shape[0]
    W[W < 1e-6] = 1e-6
    return W, N

