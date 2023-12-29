import numpy as np

from estimation.estimation_util.calcualte_standard_errors import calculate_standard_errors
from estimation.estimation_util.optimizer import estimate_parameters
from estimation.estimators.peer_effects.make_target_function_muliple_peer_networks import \
    make_target_function_multiple_peer_networks
from util.print_effective_sample_size import print_q_statistic


def estimate_model_peer_effect(y, X, peer_adj_m, initial_gamma=0.01, initial_params=None, n_buckets=100, n_runs=5,
                               consider_separate_networks=True):
    if initial_params.any() == None:
        initial_params = np.zeros(X.shape[1])

    gamma_hat = initial_gamma
    params_hat = initial_params
    gamma_hat_list = [gamma_hat]
    param_hat_list = [params_hat]

    for i in range(n_runs):

        calculate_hessian = i == n_runs - 1  # only calculate hessian only in the last run
        gamma_hat, params_hat, std_error, inverse_hessian, log_value_and_score_stable, important_sampling_weights = estimate_model_peer_effect_one_run(
            y=y, peer_adj_m=peer_adj_m,
            X=X,
            gamma_for_sampling=gamma_hat,
            params=params_hat,
            n_buckets=n_buckets,
            consider_separate_networks=consider_separate_networks,
            calclate_hessian=calculate_hessian)

        print_q_statistic(important_sampling_weights)

        gamma_hat_list.append(gamma_hat)
        param_hat_list.append(params_hat)
        print("")
        print(" the estimation results:  " + str(gamma_hat) + "  (" + (
            "NAN" if std_error == None else str(std_error[0])) + ")")
        print("")
        for a in params_hat:
            print(a, end=" ")
        print()

    info_dict = {
        "gamma_hat_list": gamma_hat_list,
        "param_hat_list": param_hat_list,
        "gamma_hat": gamma_hat,
        "param_hat": params_hat,
        "all_params_hat": np.array([gamma_hat] + list(params_hat)),
        "std_errors": std_error,
        "inverse_hessian": inverse_hessian,
        "log_likelihood_value_and_score_function": log_value_and_score_stable,
        "important_sampling_weights": important_sampling_weights
    }

    return gamma_hat, params_hat, info_dict


def estimate_model_peer_effect_one_run(y, peer_adj_m, X, gamma_for_sampling, params, n_buckets=100,
                                       consider_separate_networks=True, calclate_hessian=False):
    "Creating the target function"
    log_value_and_score_stable, _, _, important_sampling_weights = make_target_function_multiple_peer_networks(y, X,
                                                                                                               peer_adj_m,
                                                                                                               params,
                                                                                                               gamma_for_sampling,
                                                                                                               n_buckets=n_buckets,
                                                                                                               consider_separate_networks=consider_separate_networks)

    "The Optimisation"
    x0 = np.array([gamma_for_sampling] + list(params))
    optimal_params, target_value_found = estimate_parameters(log_value_and_score_stable, x0)
    gamma_hat = optimal_params[0]
    params_hat = optimal_params[1:]

    "The standard errors"
    std_error, inverse_hessian = None, None
    if (calclate_hessian):
        # log_value_and_score_stable, _, _, _ = make_target_function_multiple_peer_networks(y, X, peer_adj_m, params_hat, gamma_hat, n_buckets=n_buckets, consider_separate_networks=consider_separate_networks)
        std_error, inverse_hessian = calculate_standard_errors(log_value_and_score_stable, optimal_params)

    return gamma_hat, params_hat, std_error, inverse_hessian, log_value_and_score_stable, important_sampling_weights
