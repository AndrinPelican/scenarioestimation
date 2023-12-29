import numpy as np

from estimation.estimation_util.calcualte_standard_errors import calculate_standard_errors
from estimation.estimation_util.optimizer import estimate_parameters
from estimation.sample_buckets.buckets.bucket import Bucket
from estimation.sample_buckets.buckets.calculate_probability_of_bucket import calculate_log_probability, \
    calculate_derivative

"""
In the reciprocity case you can enumerate all the different scenario for each subgame, and thus evaluate the 

"""


def estimate_model_reciprocity_exact(adj_m, model):
    log_value_and_score_exact = make_target_function_for_model_reciprocity_exact(adj_m, model)
    x0 = model.get_initial_params()
    optimal_params, _ = estimate_parameters(log_value_and_score_exact, x0)
    gamma_hat = optimal_params[0]
    params_hat = optimal_params[1:]

    "The standard errors"
    std_error, inverse_hessian = calculate_standard_errors(log_value_and_score_exact, optimal_params)

    info_dict = {
        "gamma_hat_list": None,
        "param_hat_list": None,
        "gamma_hat": gamma_hat,
        "param_hat": params_hat,
        "all_params_hat": optimal_params,
        "std_errors": std_error,
        "inverse_hessian": inverse_hessian,
        "log_likelihood_value_and_score_function": log_value_and_score_exact,
        "important_sampling_weights": None
    }
    return gamma_hat, params_hat, info_dict


"""
This function takes the buckets, and the weights of the buckets and creates the target function.
"""


def make_target_function_for_model_reciprocity_exact(adj_m, model):
    def log_value_and_score_stable(params):
        """
        :param params: A np.array with the parameters to optimize, it contains as first value gamma,
        :return:
        """
        gamma = params[0]
        a = params[1:]
        mue = model.forward(a)

        # gamma
        if (gamma <= 0.00001):
            total_derivative = np.zeros(params.shape)
            total_derivative[0] = -100000
            return 10 ^ 30000000, total_derivative

        log_prop_total, der_gamma_total, der_mue = caluclate_prob_and_derivative_for_adj_m(adj_m, gamma, mue)
        derivative_a = model.backward(der_mue)
        derivative = np.insert(derivative_a, 0, der_gamma_total)

        # print("            params gamma:  " + str(gamma))
        # print("            der gamma:     " + str(der_gamma_total))
        # print("                                            params a :        " + str(a))
        # print("                                            der derivative_a: " + str(derivative_a))
        # print(log_prop_total)
        return - log_prop_total, - derivative

    return log_value_and_score_stable


def probability_and_derivatives_for_one_pair(mue_1, mue_2, gamma, is_one_to_two, is_two_to_one):
    mue_1_2 = np.array([mue_1, mue_2])
    if (is_one_to_two and is_two_to_one):
        bucket_a = Bucket(2, np.array([-10000, -10000]), np.array([0, 0]))
        bucket_b = Bucket(2, np.array([0, -10000]), np.array([1, 0]))
        bucket_ab = Bucket(2, np.array([-10000, 0]), np.array([0, 1]))

        # calculate probability
        log_probability_a = calculate_log_probability(bucket_a, mue_1_2, gamma)
        log_probability_b = calculate_log_probability(bucket_b, mue_1_2, gamma)
        log_probability_ab = calculate_log_probability(bucket_ab, mue_1_2, gamma)
        log_probability_total = np.log(
            np.exp(log_probability_a) + np.exp(log_probability_b) + np.exp(log_probability_ab))

        # calculate derivative
        derivative_mue_a, derivative_gamma_a = calculate_derivative(bucket_a, mue_1_2, gamma)
        derivative_mue_b, derivative_gamma_b = calculate_derivative(bucket_b, mue_1_2, gamma)
        derivative_mue_ab, derivative_gamma_ab = calculate_derivative(bucket_ab, mue_1_2, gamma)

        # adjustments:
        derivative_mue_a = derivative_mue_a * np.exp(log_probability_a)
        derivative_gamma_a = derivative_gamma_a * np.exp(log_probability_a)

        derivative_mue_b = derivative_mue_b * np.exp(log_probability_b)
        derivative_gamma_b = derivative_gamma_b * np.exp(log_probability_b)

        derivative_mue_ab = derivative_mue_ab * np.exp(log_probability_ab)
        derivative_gamma_ab = derivative_gamma_ab * np.exp(log_probability_ab)

        derivative_mue = (derivative_mue_a + derivative_mue_b + derivative_mue_ab) / np.exp(log_probability_total)
        derivative_gamma = (derivative_gamma_a + derivative_gamma_b + derivative_gamma_ab) / np.exp(
            log_probability_total)

    if (not (is_one_to_two) and not (is_two_to_one)):
        bucket = Bucket(2, np.array([0, 0]), np.array([10000, 10000]))
        # calculate probability
        log_probability_total = calculate_log_probability(bucket, mue_1_2, gamma)
        # calculate derivative
        derivative_mue, derivative_gamma = calculate_derivative(bucket, mue_1_2, gamma)

    if (is_one_to_two and not (is_two_to_one)):
        bucket = Bucket(2, np.array([-10000, 1]), np.array([0, 10000]))
        # calculate probability
        log_probability_total = calculate_log_probability(bucket, mue_1_2, gamma)
        # calculate derivative
        derivative_mue, derivative_gamma = calculate_derivative(bucket, mue_1_2, gamma)

    if (not (is_one_to_two) and is_two_to_one):
        bucket = Bucket(2, np.array([1, -10000]), np.array([10000, 0]))
        # calculate probability
        log_probability_total = calculate_log_probability(bucket, mue_1_2, gamma)
        # calculate derivative
        derivative_mue, derivative_gamma = calculate_derivative(bucket, mue_1_2, gamma)

    return log_probability_total, derivative_mue[0], derivative_mue[1], derivative_gamma


def caluclate_prob_and_derivative_for_adj_m(adj_m, gamma, mue):
    # preperation
    der_mue = np.zeros(mue.shape)
    n = adj_m.shape[0]
    index_matrix = get_index_matrix(n)
    der_gamma_total = 0
    log_prop_total = 0
    # calculation
    for i in range(n):
        for j in range(i + 1, n):
            ind_1 = index_matrix[i, j]
            ind_2 = index_matrix[j, i]
            mue_1 = mue[ind_1]
            mue_2 = mue[ind_2]
            one_to_two = int(adj_m[i, j])
            two_to_one = int(adj_m[j, i])

            log_prop, der_mue_1, der_mue_2, der_gamma = probability_and_derivatives_for_one_pair(mue_1, mue_2, gamma,
                                                                                                 one_to_two, two_to_one)
            log_prop_total += log_prop
            der_gamma_total += der_gamma
            assert der_mue[ind_1] == 0
            assert der_mue[ind_2] == 0
            der_mue[ind_1] = der_mue_1
            der_mue[ind_2] = der_mue_2

    return log_prop_total, der_gamma_total, der_mue


def get_index_matrix(n):
    index_matrix = np.zeros((n, n), dtype=int)
    k = 0
    for i in range(n):
        for j in range(n):
            if (i == j):
                continue
            index_matrix[i, j] = int(k)
            k += 1
    return index_matrix
