import numpy as np

from estimation.estimation_util.calcualte_standard_errors import calculate_standard_errors
from estimation.estimation_util.optimizer import estimate_parameters
from estimation.estimators.network.make_target_function_network import create_buckets_make_target_function_for_model


def estimate_model_network(adj_m, model, s_function, n_buckets=100, n_runs=3):
    gamma_hat = model.get_initial_params()[0]
    params_hat = model.get_initial_params()[1:]

    gamma_hat_list = [gamma_hat]
    param_hat_list = [params_hat]
    for ith_run in range(n_runs):
        gamma_hat, params_hat, std_error, inverse_hessian, log_value_and_score_stable = estimate_model_network_one_run(
            adj_m=adj_m,
            model=model,
            s_function=s_function,
            gamma_for_sampling=gamma_hat,
            initial_params=params_hat,
            n_buckets=n_buckets,
            return_std_error=ith_run == n_runs - 1)
        gamma_hat_list.append(gamma_hat)
        param_hat_list.append(params_hat)
        print("")
        print("                  the estimation results:  " + str(gamma_hat) + "  (" + str(std_error[0]) + ")")
        print("")
        for a in params_hat:
            print("%.3f" % a, end=", ")
        print()

    info_dict = {
        "gamma_hat_list": gamma_hat_list,
        "param_hat_list": param_hat_list,
        "all_params_hat": np.array([gamma_hat] + list(params_hat)),
        "std_errors": std_error,
        "inverse_hessian": inverse_hessian,
        "log_likelihood_value_and_score_function": log_value_and_score_stable
    }
    return gamma_hat, params_hat, info_dict


def estimate_model_network_one_run(adj_m, model, s_function, gamma_for_sampling, initial_params, n_buckets=100,
                                   return_std_error=True):
    "Creating the target function"
    log_value_and_score_stable, log_value_and_score, value_and_score = create_buckets_make_target_function_for_model(
        adj_m, model, s_function,
        gamma_for_sampling, initial_params,
        n_buckets)

    "The Optimisation"
    x0 = np.append(gamma_for_sampling, initial_params)
    optimal_params, _ = estimate_parameters(log_value_and_score_stable, x0)
    gamma_hat = optimal_params[0]
    a_hat = optimal_params[1:]

    "The standard errors"
    std_error = [""]
    inverse_hessian = None
    if return_std_error:
        std_error, inverse_hessian = calculate_standard_errors(log_value_and_score_stable, optimal_params)

    return gamma_hat, a_hat, std_error, inverse_hessian, log_value_and_score_stable
