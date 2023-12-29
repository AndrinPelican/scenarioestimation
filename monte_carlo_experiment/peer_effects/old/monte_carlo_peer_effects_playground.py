"""

This is a monte carlo experiment



"""

import pickle

import latextable
import matplotlib.pyplot as plt
import numpy as np
from texttable import Texttable

from estimation.estimators.peer_effects.estimate_peer_effects import estimate_model_peer_effect
from monte_carlo_experiment.monte_carlo_util.likelyhood_ratio import likelyhood_ratio
from monte_carlo_experiment.monte_carlo_util.wald_test import wald_test
from monte_carlo_experiment.peer_effects.create_frendschip_network.create_5_neighbours import \
    block_erdös_reny_reciprocal
from monte_carlo_experiment.peer_effects.create_frendschip_network.sampling_peer_decition_plain import \
    sampling_peer_decisions

gamma = 0.2
N = 500
n_buckets = 200
n_simulations = 625
# n_simulations = 225
n_runs = 2
n_blocks = 1

name = "run_27.pickle"
comment = ""

## network creation function
# adj_m_peers = peer_network_5_neighbours(n_agents=N)
# adj_m_peers = erdoes_reny_with_average_5_friends(n_agents=N)
# adj_m_peers = erdoes_reny_with_average_5_friends_recipocal(n_agents=N)
# create_network = block_erdös_reny_no_cycles
# adj_m_peers = block_marix_with_average_5_friends(n_agents=N)
create_network = block_erdös_reny_reciprocal
# create_network = block_erdös_reny
# create_network = block_random_geometric_graph

params = np.array([-1, -0.5, -1, +0.5])
# binary matrix for the two groups
X = np.zeros((N, 4))
group_1 = np.random.randint(2, size=(N))
X[:, 0] = group_1
X[:, 1] = 1 - group_1
X[:, 2] = np.random.uniform(0, 1, size=(N))
X[:, 3] = np.random.uniform(0, 1, size=(N))
"""
params = np.array([-1])
X = np.ones((N,1))
"""

gamma_hats = []
gamma_std = []
denitys = []
n_added_list = []

"""
Different configurations on how peers are connected
"""
adj_m_peers = create_network(n_agents=N, n_blocks=n_blocks)
average_degree = sum(sum(adj_m_peers)) / (N)
density = sum(sum(adj_m_peers))
reciprocity = sum(sum(adj_m_peers * np.transpose(adj_m_peers, [1, 0]))) / denitys

l_r_list = []
w_t_lsit = []
info_dict_list = []

for i in range(n_simulations):
    print("==================================================================" + str(i))
    y, n_added = sampling_peer_decisions(gamma=gamma, params=params, adj_m_peers=adj_m_peers, X=X, report_adding=True)
    print("Density:  " + str(sum(y) / N))
    denitys.append(sum(y) / (N))
    n_added_list.append(n_added)
    gamma_hat, a_hat, info_dict = estimate_model_peer_effect(y=y, X=X, peer_adj_m=adj_m_peers, initial_params=params,
                                                             initial_gamma=gamma, n_buckets=n_buckets, n_runs=n_runs,
                                                             consider_separate_networks=True)
    l_r, quantile_lr = likelyhood_ratio(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
    w_t, quantile_w = wald_test(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
    info_dict["l_r"] = quantile_lr
    info_dict["quantile_lr"] = quantile_lr
    info_dict["quantile_w"] = quantile_w
    info_dict["w_t"] = quantile_w
    info_dict["density_of_y"] = sum(y) / (N)
    info_dict["y"] = y
    info_dict["log_likelihood_value_and_score_function"] = "not included, because cannot pickle functions"
    info_dict_list.append(info_dict)

    gamma_hats.append(gamma_hat)
    gamma_std.append(info_dict["std_errors"][0])
    l_r_list.append(quantile_lr)
    w_t_lsit.append(quantile_w)

plt.hist(l_r_list)
plt.show()

plt.hist(w_t_lsit)
plt.show()

pickle_dict = {
    "info_dict_list": info_dict_list,
    "N": N,
    "params": params,
    "gamma": gamma,
    "n_buckets": n_buckets,
    "n_runs": n_runs,
    "name_network_creation": create_network.__name__,
    "n_blocks": n_blocks,
    "block_size": N / n_blocks,
    "average_degree": average_degree,
    "density": density,
    "reciprocity": reciprocity,
    "adj_m": adj_m_peers,
    "X": X,
    "comment": comment
}
file_to_save = open("./saved_simulations/" + name, 'wb')
pickle.dump(pickle_dict, file_to_save)

count_values_in_5p_range = 0
for i in range(n_simulations):
    normalized = (gamma_hats[i] - gamma) / gamma_std[i]
    if (np.abs(normalized) < 1.96):
        count_values_in_5p_range += 1

mean_gamma = np.mean(gamma_hats)
median_gamma = np.median(gamma_hats)
std_estimations = np.std(gamma_hats)
mean_estimated_std = np.mean(gamma_std)
print("std estimation " + str(np.std(gamma_hats) / np.sqrt(n_simulations)))
print("std single " + str(np.std(gamma_hats)))

table_rows = [["N = " + str(N), ""]]
table_rows.append(["simulations = " + str(n_simulations), ""])
table_rows.append(["runs = " + str(n_runs), ""])
table_rows.append(["params = " + str(params), ""])
table_rows.append(["buckets = " + str(n_buckets), "gamma"])
table_rows.append(["mean", mean_gamma])
table_rows.append(["median", median_gamma])
table_rows.append(["Std. Dev.", std_estimations])
table_rows.append(["Mean Std. Err.", mean_estimated_std])
table_rows.append(["Coverage", count_values_in_5p_range / n_simulations])

# see https://github.com/JAEarly/latextable for documentation on makeing table
table = Texttable()
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['a',
                      'a'
                      ])  # automatic
table.set_cols_align(["l", "r"])
table.add_rows(table_rows)
print(table.draw() + "\n")
print('\nTexttable Table:')
print(table.draw())
print(latextable.draw_latex(table, caption="Monte Carlo Experiment", label="table:mc_experiment") + "\n")

print("count_values_out_10p_range  " + str(count_values_in_5p_range / n_simulations))
print(gamma_std)

n_bins = 25
fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
axs.hist(gamma_hats, bins=n_bins)
axs.axvline(mean_gamma, color='g')
axs.axvline(x=gamma, color='r')
axs.set_xlabel('gamma peer effects')
plt.show()
