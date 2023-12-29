import numpy as np

from estimation.sample_buckets.network.create_digraph_for_taste_shock import create_digraph_for_taste_shock_network
from estimation.sample_buckets.sampling_util import sample_from_above, sample_from_below
from util.graph_functions.adj_matrix_array import matrix_to_array, array_to_adj_matrix


def sample_taste_shock_network(gamma_for_sampling, params, model, s_function, adj_m):
    """
    This is the create_frendschip_network algorithm form paper bucket estimation
    :param gamma:
    :param adj_m:
    :return:
    """

    # preparation
    thresholds_gamma_0 = model.forward(params)
    s_values_of_graph = s_function(adj_m)
    s_values_flatten = matrix_to_array(s_values_of_graph)
    thresholds_for_y_observed = thresholds_gamma_0 + gamma_for_sampling * s_values_flatten

    adj_m_flatten = matrix_to_array(adj_m)
    sampled_taste_shock_flatten = np.zeros(adj_m_flatten.shape)
    MN = adj_m_flatten.shape[0]
    log_importance_sampling_weight = 0

    """
        1) sample the taste shock where the decision is no!
    """
    for k in range(MN):
        if (adj_m_flatten[k] == 0):
            # no link, sample from threshold above (formed no link, despite existing links)
            sample = sample_from_above(thresholds_for_y_observed[k])
            sampled_taste_shock_flatten[k], w = sample
            log_importance_sampling_weight += np.log(w)
        else:
            # has link, put temporary to -10000 so that for sure a link is formed, this is not sampled yet
            sampled_taste_shock_flatten[k] = -10 ** 100

    """
        2) sample the taste shock where the decision is yes!
    """
    for k in range(MN):
        # print("round: "+ str(k))
        if (adj_m_flatten[k] == 0):  # were there is no edge, it already has been sampled
            continue

        else:
            # temporary evaluation of whole graph to get the current treshold
            sampled_taste_shock_flatten[
                k] = 10 ** 100  # set temporary to 1000, so that is is not formed in the next matrix
            taste_shock_realisations = array_to_adj_matrix(sampled_taste_shock_flatten)
            temporary_adj_m = create_digraph_for_taste_shock_network(taste_shock_realisations, params, model,
                                                                     gamma=gamma_for_sampling, s_function=s_function)
            """ 
            Here make sure that the temporary_adj_m is smaller then the ture (this has to be the case according
            to theory) 
            
            May occure due to numerical problems
            
            """
            if not (temporary_adj_m <= adj_m).all():
                print("The params")
                print(params)
                print(gamma_for_sampling)
                print(" we are in round: " + str(k))
                for d in sampled_taste_shock_flatten:
                    print(str(d) + ", ", end="")
                print("")
                print("")
                print("Tresholds")

                for d in thresholds_for_y_observed:
                    print(str(d) + ", ", end="")
                print("")
                assert False

            temporary_s_values_of_graph = s_function(temporary_adj_m)
            temporary_s_values_flatten = matrix_to_array(temporary_s_values_of_graph)
            temporary_thresholds = thresholds_gamma_0 + gamma_for_sampling * temporary_s_values_flatten

            sample, w = sample_from_below(temporary_thresholds[k])
            log_importance_sampling_weight += np.log(w)
            sampled_taste_shock_flatten[k] = sample

    return array_to_adj_matrix(sampled_taste_shock_flatten), log_importance_sampling_weight
