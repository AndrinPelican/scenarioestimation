import numpy as np
import copy
from util.graph_functions.adj_matrix_array import  array_to_adj_matrix


def taste_shock_realisation(n):
    return np.random.normal(loc=0.0, scale=1.0, size=(n,n))

def clear_selfloops(adj_m):
    for i in range(adj_m.shape[0]):
        adj_m[i,i]=0
    return adj_m



def create_digraph_for_taste_shock_network(taste_shock_realisations, params, model, gamma, s_function, print_info= False):

    # preparation
    thresholds_gamma_0_m = array_to_adj_matrix(model.forward(params))

    adj_m = np.array(taste_shock_realisations < thresholds_gamma_0_m, dtype=np.float64)
    adj_m = clear_selfloops(adj_m)
    adj_m_org = copy.copy(adj_m)
    adj_m_privious = np.zeros(adj_m.shape)

    while (adj_m != adj_m_privious).any():
        adj_m_privious = adj_m
        " 2) consider tranitive preferences"
        adj_m = np.array(taste_shock_realisations < thresholds_gamma_0_m + gamma * s_function(adj_m), dtype=np.float64)
        adj_m = clear_selfloops(adj_m)

    added_edges = np.sum(adj_m - adj_m_org)
    if (print_info):
        print('added edges due to transitivity preferences:   ' + str(added_edges))
    return adj_m


