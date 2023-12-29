import numpy as np

from estimation.estimation_util.make_target_function_to_model import make_target_function_for_model
from estimation.model.model_linear import LinearModel
from estimation.sample_buckets.buckets.calculate_probability_of_bucket import calculate_log_probability
from estimation.sample_buckets.peer_effects.bucket_from_shook_peereffect import bucket_from_shook_peer_effect
from estimation.sample_buckets.peer_effects.sample_taste_shock_peer_effect import sample_taste_shock_peer_effects


def create_buckets_make_target_function_for_model_peer(y, X, peer_adj_m, initial_params, gamma_for_sampling,
                                                       n_buckets=100):
    model = LinearModel(X)
    buckets = []
    bucket_weights_tilde = []
    log_important_sampling_weights = []
    # simulate 100 buckets
    for _ in range(n_buckets):
        # nearly all time used for this
        taste_shock, log_importance_sampling_weight = sample_taste_shock_peer_effects(y=y, peer_adj_m=peer_adj_m,
                                                                                      gamma_for_sampling=gamma_for_sampling,
                                                                                      params=initial_params,
                                                                                      model=model)
        bucket = bucket_from_shook_peer_effect(y=y, peer_adj_m=peer_adj_m, taste_shock=taste_shock,
                                               gamma_for_sampling=gamma_for_sampling,
                                               params=initial_params, model=model)
        buckets.append(bucket)

        # calculating the probably with which the bucket is drawn
        mue = model.forward(initial_params)
        log_p_base = calculate_log_probability(bucket, mue, gamma_for_sampling)
        p_log_tilde = log_p_base - log_importance_sampling_weight
        bucket_weights_tilde.append(p_log_tilde)
        log_important_sampling_weights.append(log_importance_sampling_weight)

    print("log_important_sampling_weights are: min " + "{0:0.1f}".format(
        min(log_important_sampling_weights)) + "   max " + "{0:0.1f}".format(
        max(log_important_sampling_weights)) + "   difference " + "{0:0.1f}".format(
        max(log_important_sampling_weights) - min(log_important_sampling_weights)) + "   std " + "{0:0.1f}".format(
        np.std(log_important_sampling_weights)))

    log_value_and_score_stable, log_value_and_score, value_and_score = make_target_function_for_model(buckets,
                                                                                                      bucket_weights_tilde,
                                                                                                      model)
    return log_value_and_score_stable, log_value_and_score, value_and_score, log_important_sampling_weights


def create_buckets_make_target_function_for_model_peer_independent(y, X, peer_adj_m, initial_params, gamma_for_sampling,
                                                                   n_buckets=100):
    log_value_and_score_stable_1, log_value_and_score, value_and_score, important_sampling_weights = create_buckets_make_target_function_for_model_peer(
        y, X, peer_adj_m, initial_params, gamma_for_sampling, n_buckets=100)
    log_value_and_score_stable_2, log_value_and_score, value_and_score, important_sampling_weights = create_buckets_make_target_function_for_model_peer(
        y, X, peer_adj_m, initial_params, gamma_for_sampling, n_buckets=100)

    def value_and_score_independent(x):
        # the two independent estimations
        neg_prob_1, neg_score_1 = log_value_and_score_stable_1(x)
        neg_prob_2, neg_score_2 = log_value_and_score_stable_2(x)

        # neg_prob == 0 is when gammy <= 0 This case is not allowed
        if (neg_prob_2 == 0):
            print("got negative gamma")
            total_derivative = np.zeros(x.shape)
            total_derivative[0] = -100000
            return 10 ^ 30000000, total_derivative
        # print("                              "+str(-neg_prob_1)+",")

        log_score = (1 / -neg_prob_2) * neg_score_1
        return 0, log_score

    return value_and_score_independent, None, None, important_sampling_weights
