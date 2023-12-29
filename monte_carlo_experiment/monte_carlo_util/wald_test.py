
import numpy as np
from scipy.stats import chi2




def wald_test(params_null, info_dict):
    """
    See:  https://de.wikipedia.org/wiki/Wald-Test
    """
    all_params_hat = info_dict["all_params_hat"]
    d_params = params_null - all_params_hat
    inverse_hessian = info_dict["inverse_hessian"]
    w_stat = np.inner(d_params, np.dot(np.linalg.inv(inverse_hessian), d_params))
    quantile = chi2.cdf(w_stat, df = len(params_null))
    return w_stat, quantile