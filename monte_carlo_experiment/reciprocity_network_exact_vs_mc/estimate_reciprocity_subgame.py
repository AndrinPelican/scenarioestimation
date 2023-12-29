"""
This file contains different methods to estimate the 2X2 sub-game for the reciprocity case


Note: the estimated probaiblity from one bucket can be bigger then 1


Working with only one bucket sometimes makes optimisation not working, figure out what the problem is!

"""

import numpy as np

from estimation.estimators.network.estimate_model_network import create_buckets_make_target_function_for_model


def estimate_gradient_and_probability_independent(adj_m, model, s_function, from_shock_to_bucket_monotone_increasing,
                                                  gamma_for_sampling, initial_params, n_buckets):
    _, _, value_and_score_1 = create_buckets_make_target_function_for_model(
        adj_m, model, s_function, from_shock_to_bucket_monotone_increasing, gamma_for_sampling, initial_params,
        n_buckets)
    _, _, value_and_score_2 = create_buckets_make_target_function_for_model(
        adj_m, model, s_function, from_shock_to_bucket_monotone_increasing, gamma_for_sampling, initial_params,
        n_buckets)

    def value_and_score_independent(x):
        # the two independent estimations
        neg_prob_1, neg_score_1 = value_and_score_1(x)
        neg_prob_2, neg_score_2 = value_and_score_2(x)

        # neg_prob == 0 is when gammy <= 0 This case is not allowed
        if (neg_prob_2 == 0):
            print("got negative gamma")
            total_derivative = np.zeros(x.shape)
            total_derivative[0] = -100000
            return 10 ^ 30000000, total_derivative
        # print("                              "+str(-neg_prob_1)+",")

        log_score = (1 / -neg_prob_2) * neg_score_1
        return 0, log_score

    return value_and_score_independent


"""

In this variant we apply bias correction as in 

https://math.stackexchange.com/questions/2297482/is-there-an-unbiased-estimator-of-the-reciprocal-of-the-slope-in-linear-regressi
"""


def estimate_gradient_and_probability_independent_bias_correction(adj_m, model, s_function, gamma_for_sampling,
                                                                  initial_params, n_buckets):
    estimators_for_reciprocal_probability = []
    for _ in range(2):
        _, _, value_and_score_1 = create_buckets_make_target_function_for_model(
            adj_m, model, s_function, gamma_for_sampling, initial_params, n_buckets=5)
        estimators_for_reciprocal_probability.append(value_and_score_1)

    _, _, value_and_score_2 = create_buckets_make_target_function_for_model(
        adj_m, model, s_function, gamma_for_sampling, initial_params,
        n_buckets)

    def value_and_score_independent(x):
        # the two independent estimations
        probs_for_scling_constant = []
        for value_and_score in estimators_for_reciprocal_probability:
            neg_prob, neg_score = value_and_score(x)
            probs_for_scling_constant.append(-neg_prob)

        mean_prob = np.mean(probs_for_scling_constant)
        var_prob = np.var(probs_for_scling_constant)
        invers_estimate = 1 / mean_prob - var_prob / (mean_prob ** 3)
        # print("                "+ str(- var_prob/(mean_prob**3))+"         " + str(1/mean_prob)+"         " +str(invers_estimate))

        neg_prob_2, neg_score_2 = value_and_score_2(x)
        # neg_prob == 0 is when gammy <= 0 This case is not allowed
        if (neg_prob_2 == 0):
            print("got negative gamma")
            total_derivative = np.zeros(x.shape)
            total_derivative[0] = -100000
            return 10 ^ 30000000, total_derivative

        log_score = invers_estimate * neg_score_2
        # print("                        "+str(mean_prob))

        return 0, log_score

    return value_and_score_independent


"""

In this variant we apply the unbiased estimator like in:

Unbiased Estimation of the Reciprocal Mean for Non-negative Random Variables
https://arxiv.org/pdf/1907.01843.pdf

"""


def estimate_gradient_and_probability_independent_unbiased(adj_m, model, s_function,
                                                           from_shock_to_bucket_monotone_increasing, gamma_for_sampling,
                                                           initial_params, n_buckets):
    estimators_for_reciprocal_probability = []

    # See paper to see what p has to be set and
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html
    # w =0.05
    # p = 0.01861516453443046
    w = 2
    p = 0.5

    k = np.random.geometric(p=p)
    n = k - 1
    q_n = p * (1 - p) ** n
    for _ in range(n):
        _, _, value_and_score_1 = create_buckets_make_target_function_for_model(
            adj_m, model, s_function, from_shock_to_bucket_monotone_increasing, gamma_for_sampling, initial_params,
            n_buckets=4)
        estimators_for_reciprocal_probability.append(value_and_score_1)

    _, _, value_and_score_1 = create_buckets_make_target_function_for_model(
        adj_m, model, s_function, from_shock_to_bucket_monotone_increasing, gamma_for_sampling, initial_params,
        n_buckets)

    def value_and_score_independent(x):
        nnn = n
        neg_prob_1, neg_score_1 = value_and_score_1(x)
        # neg_prob == 0 is when gammy <= 0 This case is not allowed
        if (neg_prob_1 == 0):
            print("got negative gamma")
            total_derivative = np.zeros(x.shape)
            total_derivative[0] = -100000
            return 10 ^ 30000000, total_derivative
        # the two independent estimations
        inverse_estimate = w / q_n
        for value_and_score in estimators_for_reciprocal_probability:
            neg_prob, neg_score = value_and_score(x)
            Z = -neg_prob
            inverse_estimate = inverse_estimate * (1 - w * Z)
        #     print(".",end="")
        # print("")
        print(" should be around 0.33                " + str(1 / inverse_estimate))

        log_score = inverse_estimate * neg_score_1
        # print("                        "+str(mean_prob))

        return 0, log_score

    return value_and_score_independent
