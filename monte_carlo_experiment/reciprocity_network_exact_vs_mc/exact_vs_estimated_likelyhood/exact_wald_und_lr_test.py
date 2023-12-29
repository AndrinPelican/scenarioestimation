"""

Scenario Estimation is a random estimation methods, because the scenarios are sampled.

This file has as purpose to see how this random is in comparison to the exact estimator.

We use Reciprocity in nyakatoke, because we can evaluate the likelihood closed form.

"""
import pickle

import matplotlib.pyplot as plt
import numpy as np

from monte_carlo_experiment.reciprocity_network_exact_vs_mc.estimate_model_reciprocity_exact import \
    estimate_model_reciprocity_exact
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from monte_carlo_experiment.monte_carlo_util.likelyhood_ratio import likelyhood_ratio
from monte_carlo_experiment.monte_carlo_util.wald_test import wald_test
from util.graph_functions.random_digraph import create_digraph_reciprocity

n = 20
# True parameters
params = np.array([-0.2])
gamma = 0.5
n_estimations = 1000

erdoes_reny_model = ErdoesRenyModel(n)

gamma_hats_1_bucket = []
gamma_hats_10_bucket = []
gamma_subgames_indpendent = []
gamma_subgames_indpendent_biased_corrected = []
gamma_subgames_indpendent_unbiased = []

l_r_list = []
w_t_lsit = []
info_dict_list = []

for i in range(n_estimations):
    print("====================================================================   " + str(i))

    adj_m = create_digraph_reciprocity(n, params=params, model=erdoes_reny_model, gamma=gamma)
    gamma_hat_exact, a_hat, info_dict = estimate_model_reciprocity_exact(adj_m=adj_m, model=erdoes_reny_model)

    l_r, quantile_lr = likelyhood_ratio(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
    w_t, quantile_w = wald_test(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
    info_dict["l_r"] = quantile_lr
    info_dict["quantile_lr"] = quantile_lr
    info_dict["quantile_w"] = quantile_w
    info_dict["w_t"] = quantile_w
    info_dict["log_likelihood_value_and_score_function"] = "not included, because cannot pickle functions"
    info_dict_list.append(info_dict)

    l_r_list.append(quantile_lr)
    w_t_lsit.append(quantile_w)

file_to_save = open("infoDicts", 'wb')
pickle.dump(info_dict_list, file_to_save)

plt.hist(l_r_list)
plt.show()

plt.hist(w_t_lsit)
plt.show()
