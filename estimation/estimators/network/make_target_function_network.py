from estimation.estimation_util.make_target_function_to_model import make_target_function_for_model
from estimation.sample_buckets.buckets.calculate_probability_of_bucket import calculate_log_probability
from estimation.sample_buckets.network.from_taste_shock_to_bucket import from_shock_to_bucket_considering_noLink
from estimation.sample_buckets.network.sample_taste_shock import sample_taste_shock_network
from util.graph_functions.adj_matrix_array import matrix_to_array

from util.print_effective_sample_size import print_q_statistic


def create_buckets_make_target_function_for_model(adj_m, model, s_function, gamma_for_sampling, initial_params,
                                                  n_buckets=100):
    buckets = []
    bucket_weights_tilde = []
    important_sampling_weights = []

    # simulate 100 buckets
    for _ in range(n_buckets):
        # print("debug: start semplind shock")
        taste_shock_1, log_importance_sampling_weight = sample_taste_shock_network(gamma_for_sampling,
                                                                                   params=initial_params,
                                                                                   model=model, s_function=s_function,
                                                                                   adj_m=adj_m)

        taste_shock_1_flatten = matrix_to_array(taste_shock_1)
        bucket = from_shock_to_bucket_considering_noLink(gamma_for_sampling, model, initial_params,
                                                         taste_shock_1_flatten, adj_m, s_function)
        buckets.append(bucket)

        # calculating the probably with which the bucket is drawn
        mue = model.forward(initial_params)
        log_p_base = calculate_log_probability(bucket, mue, gamma_for_sampling)
        p_log_tilde = log_p_base - log_importance_sampling_weight
        bucket_weights_tilde.append(p_log_tilde)
        important_sampling_weights.append(log_importance_sampling_weight)

    print_q_statistic(important_sampling_weights)

    log_value_and_score_stable, log_value_and_score, value_and_score = make_target_function_for_model(buckets,
                                                                                                      bucket_weights_tilde,
                                                                                                      model)
    return log_value_and_score_stable, log_value_and_score, value_and_score
