import numpy as np

from estimation.sample_buckets.sampling_util import sample_from_above, sample_from_below
from monte_carlo_experiment.peer_effects.create_frendschip_network.sampling_peer_decition_plain import \
    derive_desicions_from_taste_shock_peer_effects

"""
The mat mul is the time consuming part of the whole process

"""


def sample_taste_shock_peer_effects(y, peer_adj_m, gamma_for_sampling, params, model):
    """
    This is the create_frendschip_network algorithm form paper bucket estimation
    :param gamma:
    :param adj_m:
    :return:
    """

    thresholds_gamma_0 = model.forward(params)

    # preparation
    N = y.shape[0]
    thresholds_for_y_observed = thresholds_gamma_0 + gamma_for_sampling * np.matmul(peer_adj_m, y)
    sampled_taste_shock = np.zeros(y.shape)
    log_importance_sampling_weight = 0

    """
        1) sample the taste shock where the decision is no!
    """
    for k in range(N):
        if (y[k] == 0):
            # no link, sample from threshold above (formed no link, despite existing links)
            sample = sample_from_above(thresholds_for_y_observed[k])
            sampled_taste_shock[k], w = sample
            log_importance_sampling_weight += np.log(w)
        else:
            # has link, put temporary to -10000 so that for sure a link is formed, this is not sampled yet
            sampled_taste_shock[k] = -10000

    """
        2) sample the taste shock where the decision is yes!
    """
    # sample where there are links
    for k in range(N):

        # where the decision is no, we already sampled
        if (y[k] == 0):
            continue

        else:
            # 1) Determining threshold
            # temporary evaluation of whole graph to get the current threshold
            sampled_taste_shock[k] = 1000  # set temporary to 1000, so that is is not formed in the next matrix
            # t0 = time.clock()
            y_temporary, _ = derive_desicions_from_taste_shock_peer_effects(N, gamma_for_sampling, thresholds_gamma_0,
                                                                            peer_adj_m, sampled_taste_shock)
            # t1 = time.clock()
            thresholds_for_y_temporary = thresholds_gamma_0 + gamma_for_sampling * np.matmul(peer_adj_m, y_temporary)

            # 2) Sampling and book keeping
            sample, w = sample_from_below(thresholds_for_y_temporary[k])
            log_importance_sampling_weight += np.log(w)
            sampled_taste_shock[k] = sample
            # t2 = time.clock()
            # print("buckets: " + str(t1 - t0))
            # print("opt: " + str(t2 - t1))
    return sampled_taste_shock, log_importance_sampling_weight
