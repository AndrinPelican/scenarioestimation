import numpy as np


def s_reciprocity(adj_m):
    return np.transpose(adj_m, [1, 0])


def s_transitivity(adj_m):
    tansitivity_m = np.matmul(adj_m, adj_m)
    return tansitivity_m - np.diag(np.diag(tansitivity_m))  # remove the diagonal


def s_support(adj_m):
    support_m = np.matmul(np.transpose(adj_m, [1, 0]), adj_m)
    return support_m - np.diag(np.diag(support_m))  # remove the diagonal
