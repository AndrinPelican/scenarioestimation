"""
This files help to make

params and the X matrix

for a nyakatoke where we simply consider the in and out degrees


"""

import numpy as np


def params_from_indegrees_and_outdegrees(indegree_list, out_degree_list):
    return np.array(indegree_list+out_degree_list)

def X_matrix_fixed_effect_for_network_n_agents(n):

    # preperation
    X = np.zeros((n*(n-1),n*2))
    k = 0
    # filling it up
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            X[k,i] = 1
            X[k,j+n] = 1
            k += 1

    # take out one column to avoid multicollinearity
    return X[:,1:]


