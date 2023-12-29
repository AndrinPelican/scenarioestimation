import matplotlib.pyplot as plt
import numpy as np

from estimation.estimators.network.estimate_model_network import estimate_model_network
from estimation.model.model_linear import LinearModel
from estimation.sample_buckets.network.from_taste_shock_to_bucket import \
    from_shock_to_bucket_transitivity_considering_noLink
from util.graph_functions.random_digraph import create_digraph_transitivity
from util.graph_functions.s_functions import s_transitivity
from util.params_and_X_for_degree_heterogeneity import params_from_indegrees_and_outdegrees, \
    X_matrix_fixed_effect_for_network_n_agents

"""
This is not working well, probably due to too many variables

gamma seems to be downwards biased. 

See comments in readme.
"""

n = 50
gamma = 0.03
small_a = -0.0
indegree_list = [small_a] * int(n)
outdegree_list = [small_a] * int(n)

params = params_from_indegrees_and_outdegrees(indegree_list, outdegree_list)
X = X_matrix_fixed_effect_for_network_n_agents(n)
model = LinearModel(X)
model.initial_params = params
model.initial_gamma = gamma
model.initial_gamma = 0.01

estimate_with_buckets = True

a_hats = []
gamma_hats = []
denitys = []
for i in range(2):
    print("==================================================================" + str(i))
    adj_m = create_digraph_transitivity(n, params=params, model=model, gamma=gamma)
    denitys.append(sum(sum(adj_m)) / (n * (n - 1)))

    gamma_hat, a_hat, info_dict = estimate_model_network(adj_m=adj_m,
                                                         model=model,
                                                         from_shock_to_bucket_monotone_increasing=from_shock_to_bucket_transitivity_considering_noLink,
                                                         s_function=s_transitivity,
                                                         n_buckets=1,
                                                         n_runs=3)
    print("gammas: " + str(info_dict["gamma_hat_list"]))

    gamma_hats.append(gamma_hat)
    a_hats.append(np.mean(a_hat))

print("done")
print(np.mean(gamma_hats))
print(np.mean(a_hats))
print(np.mean(denitys))

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
n_bins = 20
# We can set the number of bins with the `bins` kwarg
axs[0].hist(gamma_hats, bins=n_bins)
axs[0].axvline(x=gamma, color='r')
axs[0].set_xlabel('gamma')
axs[1].hist(a_hats, bins=n_bins)
axs[1].axvline(x=params[0], color='r')
axs[1].set_xlabel('a')
axs[2].hist(denitys, bins=n_bins)
axs[2].set_xlabel('denitys')

plt.show()
