"""

Scenario Estimation is a random estimation methods, because the scenarios are sampled.

This file has as purpose to see how this random is in comparison to the exact estimator.

We use Reciprocity in nyakatoke, because we can evaluate the likelihood closed form.

"""

import matplotlib.pyplot as plt
import numpy as np

from estimation.estimation_util.calcualte_standard_errors import calculate_standard_errors
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from monte_carlo_experiment.reciprocity_network_exact_vs_mc.estimate_model_reciprocity_considering_subgames import \
    estimate_reciprocyty_subgames
from monte_carlo_experiment.reciprocity_network_exact_vs_mc.estimate_model_reciprocity_exact import \
    estimate_model_reciprocity_exact

n = 8
# True parameters
params = np.array([-0.2])
gamma = 0.5
n_estimations = 20

erdoes_reny_model = ErdoesRenyModel(n)

gamma_hats_1_bucket = []
gamma_hats_10_bucket = []
gamma_subgames_indpendent = []
gamma_subgames_indpendent_biased_corrected = []
gamma_subgames_indpendent_unbiased = []

"""
The adj matrix has been created with:
    adj_m = create_digraph_reciprocity(n, params= params, model=erdoes_reny_model, gamma=gamma)

in order to ensure reproducibility, we wrote the simulated matrix directly into the code.
"""
adj_m = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])

# adj_m = create_digraph_reciprocity(n, params= params, model=erdoes_reny_model, gamma=gamma)

gamma_hat_exact, params_hat, info_dict = estimate_model_reciprocity_exact(adj_m=adj_m, model=erdoes_reny_model)
all_params = np.array([gamma_hat_exact] + list(params_hat))


def make_value_from_dict(info_dict):
    log_likelihood_value_and_score_function = info_dict['log_likelihood_value_and_score_function']
    "The standard errors"
    std_error, inverse_hessian = calculate_standard_errors(log_likelihood_value_and_score_function, all_params)
    return inverse_hessian[0, 0]
    # return log_likelihood_value_and_score_function(all_params)[1][1]


correct_value = make_value_from_dict(info_dict)

values = []

for i in [1, 1, 10, 10, 40, 40, 100, 100, 400]:
    gamma_hat, a_hat, info_dict = estimate_reciprocyty_subgames(adj_m=adj_m, n_buckets=i, n_runs=1,
                                                                type="not_indpendent")
    values.append(make_value_from_dict(info_dict))

plt.axhline(y=correct_value, color='r', linestyle='-')

plt.plot(values)
plt.show()
