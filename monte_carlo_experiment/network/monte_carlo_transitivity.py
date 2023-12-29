"""
TBD

"""

import matplotlib.pyplot as plt
import numpy as np

from estimation.estimators.network.estimate_model_network import estimate_model_network
from estimation.model.model_linear import LinearModel
from estimation.sample_buckets.network.from_taste_shock_to_bucket import \
    from_shock_to_bucket_transitivity_considering_noLink
from monte_carlo_experiment.monte_carlo_util.likelyhood_ratio import likelyhood_ratio
from monte_carlo_experiment.monte_carlo_util.wald_test import wald_test
from monte_carlo_experiment.monte_carlo_util.x_for_proximity_and_two_groups import proximity_and_two_groups
from util.graph_functions.random_digraph import create_digraph_transitivity
from util.graph_functions.s_functions import s_transitivity

"""

n_buckets = 5
n_simulations = 400
n_runs = 1
n = 30

"""
n_buckets = 10
n_simulations = 100
n_runs = 1
n = 30

gamma = 0.04
params = np.array([-0.2, -0.4, -0.3])

X = proximity_and_two_groups(n)
model = LinearModel(X)
model.initial_params = params
model.initial_gamma = gamma

a_hats = []
gamma_hats = []
gamma_std = []
denitys = []
l_r_list = []
w_t_lsit = []

for i in range(n_simulations):
    print("==================================================================" + str(i))
    adj_m = create_digraph_transitivity(n, params=params, model=model, gamma=gamma)
    denitys.append(sum(sum(adj_m)) / (n * (n - 1)))

    gamma_hat, a_hat, info_dict = estimate_model_network(adj_m=adj_m,
                                                         model=model,
                                                         from_shock_to_bucket_monotone_increasing=from_shock_to_bucket_transitivity_considering_noLink,
                                                         s_function=s_transitivity,
                                                         n_buckets=n_buckets,  # should be 30
                                                         n_runs=n_runs)

    print("gammas: " + str(info_dict["gamma_hat_list"]))
    gamma_hats.append(gamma_hat)
    gamma_std.append(info_dict["std_errors"][0])
    a_hats.append(a_hat[0])

    l_r, quantile = likelyhood_ratio(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
    w_t, quantile_w = wald_test(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
    l_r_list.append(quantile)
    w_t_lsit.append(quantile_w)

plt.hist(l_r_list)
plt.show()

plt.hist(w_t_lsit)
plt.show()

mean_gamma = np.mean(gamma_hats)
print("gamma, medan mean")
print(np.median(gamma_hats))
print(mean_gamma)
print("std estimation " + str(np.std(gamma_hats) / np.sqrt(n_simulations)))
print("std single " + str(np.std(gamma_hats)))
print("standarderrors, calculated from mle")
print(np.mean(gamma_std))

count_values_out_10p_range = 0
for i in range(n_simulations):
    normalized = (gamma_hats[i] - gamma) / gamma_std[i]
    if (np.abs(normalized) > 1.65):
        count_values_out_10p_range += 1

print("count_values_out_10p_range  " + str(count_values_out_10p_range / n_simulations))
print(gamma_std)

n_bins = 25
fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
axs.hist(gamma_hats, bins=n_bins)
axs.axvline(mean_gamma, color='g')
axs.axvline(x=gamma, color='r')
axs.set_xlabel('gamma transitivity')
plt.show()

# fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
# n_bins = 20
# # We can set the number of bins with the `bins` kwarg
# axs[0].hist(gamma_hats, bins=n_bins)
# axs[0].axvline(x=gamma, color='r')
# axs[0].set_xlabel('gamma')
#
#
# axs[1].hist(a_hats, bins=n_bins)
# axs[1].axvline(x=params[0], color='r')
# axs[1].set_xlabel('a')
#
# axs[2].hist(denitys, bins=n_bins)
# axs[2].set_xlabel('denitys')
#
#
# plt.show()
