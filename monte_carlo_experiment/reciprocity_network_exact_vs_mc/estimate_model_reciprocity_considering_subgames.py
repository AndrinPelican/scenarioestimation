"""

This file treats the reciprocity case, where whe consider the different subgames

For each subgame there is a derivative calculation and a log probability

Note we caluclate the gradient for each subgame and aggregate the gradient
because the gradient is calculated using different estimation, we cannot closed
give the value function. therefore we minimize gradient_transposed*Gradient


In optimisation the gradient does not conicide perfectly correspond to the target function.
This makes the gradient getting getting stuck. One way around is to use nelder mead on
gradient_transpose*gradient

"""
import numpy as np

from estimation.estimation_util.optimizer import estimate_parameters
from estimation.estimators.network.estimate_model_network import create_buckets_make_target_function_for_model
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from monte_carlo_experiment.reciprocity_network_exact_vs_mc.estimate_reciprocity_subgame import \
    estimate_gradient_and_probability_independent, estimate_gradient_and_probability_independent_bias_correction, \
    estimate_gradient_and_probability_independent_unbiased
from util.graph_functions.s_functions import s_reciprocity

"""
Making the target function and then optimize
"""


def estimate_reciprocyty_subgames(adj_m, n_buckets, n_runs, type):
    model = ErdoesRenyModel(2)
    gamma_hat = model.get_initial_params()[0]
    params_hat = model.get_initial_params()[1:]

    gamma_hat_list = [gamma_hat]
    param_hat_list = [params_hat]
    for _ in range(n_runs):
        gamma_hat, params_hat, log_value_and_score_subgames = estimate_model_reciprocity_subgames_one_run(adj_m=adj_m,
                                                                                                          gamma_for_sampling=gamma_hat,
                                                                                                          initial_params=params_hat,
                                                                                                          n_buckets=n_buckets,
                                                                                                          type=type)
        gamma_hat_list.append(gamma_hat)
        param_hat_list.append(params_hat)
        print(" the estimation results:  ", end="")

    info_dict = {
        "gamma_hat_list": gamma_hat_list,
        "param_hat_list": param_hat_list,
        "log_likelihood_value_and_score_function": log_value_and_score_subgames
    }
    return gamma_hat, params_hat, info_dict


def estimate_model_reciprocity_subgames_one_run(adj_m, gamma_for_sampling, initial_params, n_buckets, type):
    log_value_and_score_subgames = make_target_function_for_model_reciprocity_subgames(adj_m, gamma_for_sampling,
                                                                                       initial_params, n_buckets, type)
    x0 = ErdoesRenyModel(2).get_initial_params()
    optimal_params3, _ = estimate_parameters(log_value_and_score_subgames, x0)
    # optimal_params3, _ = estimate_parameters_no_grdient(log_value_and_score_subgames, x0)
    gamma_hat = optimal_params3[0]
    a_hat = optimal_params3[1:]
    return gamma_hat, a_hat, log_value_and_score_subgames


"""

split up problem to different subgames

For each create one score_and_value function

Aggregate the function to one score_and_value_function
"""


def make_target_function_for_model_reciprocity_subgames(adj_m, gamma_for_sampling, initial_params, n_buckets, type):
    n = adj_m.shape[0]
    value_and_score_functions_subgames = []

    # calculation
    for i in range(n):
        for j in range(i + 1, n):
            # make the adj matrix for sub game
            adj_m_subgame = np.array([[0, adj_m[i, j]], [adj_m[j, i], 0]])

            # score and value sub game
            value_and_score_function_subgame = make_value_and_score_function_for_subgame(adj_m_subgame,
                                                                                         gamma_for_sampling,
                                                                                         initial_params, n_buckets,
                                                                                         type)
            value_and_score_functions_subgames.append(value_and_score_function_subgame)

    def value_and_score_function(x):
        score = 0
        value = 0
        for current_value_and_score_function in value_and_score_functions_subgames:
            current_value, current_score = current_value_and_score_function(x)
            score += current_score
            value += current_value
        score = score
        value = value
        # value = np.linalg.norm(score)
        # print(value)
        # print(score)
        return value, score

    return value_and_score_function


"""

Here we implement the logic of getting the value and the score for one subgame.

"""


def make_value_and_score_function_for_subgame(adj_m, gamma_for_sampling, initial_params, n_buckets, type):
    model = ErdoesRenyModel(2)
    s_function = s_reciprocity
    value_and_score_subgame = None

    if ("not_indpendent" == type):
        log_value_and_score_stable, log_value_and_score, value_and_score = create_buckets_make_target_function_for_model(
            adj_m, model, s_function, gamma_for_sampling, initial_params,
            n_buckets)
        return log_value_and_score_stable

    if ("indpendent" == type):
        return estimate_gradient_and_probability_independent(adj_m, model, s_function,
                                                             from_shock_to_bucket_monotone_increasing,
                                                             gamma_for_sampling, initial_params, n_buckets)

    if ("indpendent_biased_corrected" == type):
        return estimate_gradient_and_probability_independent_bias_correction(adj_m, model, s_function,
                                                                             from_shock_to_bucket_monotone_increasing,
                                                                             gamma_for_sampling, initial_params,
                                                                             n_buckets)

    if ("indpendent_unbiased" == type):
        return estimate_gradient_and_probability_independent_unbiased(adj_m, model, s_function,
                                                                      from_shock_to_bucket_monotone_increasing,
                                                                      gamma_for_sampling, initial_params, n_buckets)

    return value_and_score_subgame
