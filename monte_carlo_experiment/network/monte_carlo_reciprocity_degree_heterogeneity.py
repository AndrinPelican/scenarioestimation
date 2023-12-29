import matplotlib.pyplot as plt
import numpy as np

from estimation.estimators.network.estimate_model_network import estimate_model_network
from estimation.model.model_linear import LinearModel
from monte_carlo_experiment.reciprocity_network_exact_vs_mc.estimate_model_reciprocity_exact import \
    estimate_model_reciprocity_exact
from util.graph_functions.random_digraph import create_digraph_reciprocity
from util.graph_functions.s_functions import s_reciprocity
from util.params_and_X_for_degree_heterogeneity import params_from_indegrees_and_outdegrees, \
    X_matrix_fixed_effect_for_network_n_agents

"""
This files inquires behaivor of szenario estimation when there are a lot of parameters (the degree


"""

n = 30
gamma = 0.5
small_a = -0.0
indegree_list = [small_a] * int(n)
outdegree_list = [small_a] * int(n - 1)  # -1 to avoid mulitcoliniarity

params = params_from_indegrees_and_outdegrees(indegree_list, outdegree_list)
X = X_matrix_fixed_effect_for_network_n_agents(n)
model = LinearModel(X)
model.initial_params = params
model.initial_gamma = gamma

estimate_with_buckets = True

a_hats = []
gamma_hats = []
denitys = []
for i in range(10):
    print("==================================================================" + str(i))
    adj_m = create_digraph_reciprocity(n, params=params, model=model, gamma=gamma)
    denitys.append(sum(sum(adj_m)) / (n * (n - 1)))

    if estimate_with_buckets:
        gamma_hat, a_hat, info_dict = estimate_model_network(adj_m=adj_m, model=model,
                                                             s_function=s_reciprocity, n_buckets=10, n_runs=2)
        print("gammas: " + str(info_dict["gamma_hat_list"]))
    else:
        gamma_hat, a_hat, info_dict = estimate_model_reciprocity_exact(adj_m=adj_m, model=model)
        print("gammas: " + str(gamma_hat))

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
