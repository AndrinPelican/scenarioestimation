

from scipy.stats import chi2



def likelyhood_ratio(params_null, info_dict):
    log_value_and_score_stable = info_dict["log_likelihood_value_and_score_function"]
    all_params_hat = info_dict["all_params_hat"]

    l_hat_hat, _ = log_value_and_score_stable(all_params_hat)
    l_hat_null, _ = log_value_and_score_stable(params_null)

    l_r = 2*(l_hat_null - l_hat_hat)
    quantile = chi2.cdf(l_r, df = len(params_null))
    return l_r, quantile


















