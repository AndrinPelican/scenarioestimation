from estimation.sample_buckets.network.create_digraph_for_taste_shock import taste_shock_realisation, \
    create_digraph_for_taste_shock_network
from util.graph_functions.s_functions import s_reciprocity, s_transitivity


def create_digraph_reciprocity(n, params, model, gamma):
    """
    :param n: size of adj_m (number of agents)
    :param a: parameter for density (scaler)
    :param gamma: strategic parameter
    :return:
    """

    taste_shock_realisations = taste_shock_realisation(n)
    return create_digraph_for_taste_shock_network(taste_shock_realisations, params, model, gamma,
                                                  s_function=s_reciprocity, print_info=True)


def create_digraph_transitivity(n, params, model, gamma):
    """
    :param n: size of adj_m (number of agents)
    :param a: parameter for density (scaler)
    :param gamma: strategic parameter
    :return:
    """
    taste_shock_realisations = taste_shock_realisation(n)
    return create_digraph_for_taste_shock_network(taste_shock_realisations, params, model, gamma,
                                                  s_function=s_transitivity, print_info=True)


"""
Fixme: this below is cumbersome, work direcly with with model and tresholds


def create_digraph__homopholy(n, a_in_group, a_cross_group, gamma, print_info=False):

    assert n%2==0

    taste_shock_realisations = taste_shock_realisation(n)
    n = taste_shock_realisations.shape[0]

    adj_m_threshold = np.zeros(taste_shock_realisations.shape)
    for i in range(n):
        for j in range(n):
            if (i<n/2 and j < n/2) or (i>=n/2 and j >= n/2):
                adj_m_threshold[i,j] = a_in_group
            else:
                adj_m_threshold[i,j] = a_cross_group


    adj_m = np.array(taste_shock_realisations < adj_m_threshold, dtype=np.float64)
    adj_m = clear_selfloops(adj_m)
    adj_m_org = copy.copy(adj_m)

    " 2) consider reciprocity preferences"
    adj_m = np.array(taste_shock_realisations < adj_m_threshold + gamma * s_reciprocity(adj_m), dtype=np.float64)
    adj_m = clear_selfloops(adj_m)

    added_edges = np.sum(adj_m - adj_m_org)
    if (print_info):
        print('added edges due to transitivity preferences:   ' + str(added_edges))
    return adj_m
    
"""
