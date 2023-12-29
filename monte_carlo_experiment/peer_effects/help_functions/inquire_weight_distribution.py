import numpy as np

from estimation.estimators.peer_effects import estimate_model_peer_effect
from estimation.model.model_linear import LinearModel
from monte_carlo_experiment.monte_carlo_util.likelyhood_ratio import likelyhood_ratio
from monte_carlo_experiment.monte_carlo_util.wald_test import wald_test
from monte_carlo_experiment.peer_effects.create_frendschip_network.create_5_neighbours import \
    erdoes_reny_with_average_5_friends_recipocal
from monte_carlo_experiment.peer_effects.create_frendschip_network.sampling_peer_decition_plain import \
    sampling_peer_decisions

gamma = 0.4
N = 200
n_buckets = 300
n_simulations = 200  # 625
n_runs = 2

params = np.array([-1])
X = np.random.uniform(0.2, 2, size=(N))

# This is to figure out how important the order of Y is for the weights.
# Hypothesised: the base probability of y=1 matters when ordering
# seems that reversed ture works better

X = np.array(sorted(X, reverse=True))
X = np.transpose([X], [1, 0])

model = LinearModel(X)
model.initial_gamma = gamma
model.initial_params = params

gamma_hats = []
gamma_std = []
denitys = []
n_added_list = []

"""
Different configurations on how peers are connected
"""
# adj_m_peers = peer_network_5_neighbours(n_agents=N)
# adj_m_peers = erdoes_reny_with_average_5_friends(n_agents=N)
# adj_m_peers = erdoes_reny_with_average_5_friends_no_circle(n_agents=N)
# adj_m_peers = block_marix_with_average_5_friends(n_agents=N)
# adj_m_peers = random_geometric_graph(n_agents=N, sort = False)


l_r_list = []
w_t_lsit = []
info_dict_list = []

for i in range(n_simulations):
    adj_m_peers = erdoes_reny_with_average_5_friends_recipocal(n_agents=N)

    print("average degree:  " + str(sum(sum(adj_m_peers)) / (N)))

    print("==================================================================" + str(i))

    y, n_added = sampling_peer_decisions(gamma=gamma, params=params, adj_m_peers=adj_m_peers, model=model,
                                         report_adding=True)
    print("Density:  " + str(sum(y) / N))
    denitys.append(sum(y) / (N))
    n_added_list.append(n_added)

    gamma_hat, a_hat, info_dict = estimate_model_peer_effect(y=y, peer_adj_m=adj_m_peers, model=model,
                                                             n_buckets=n_buckets, n_runs=n_runs)

    l_r, quantile_lr = likelyhood_ratio(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
    w_t, quantile_w = wald_test(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
    info_dict["quantile_lr"] = quantile_lr
    info_dict["quantile_w"] = quantile_w
    info_dict["log_likelihood_value_and_score_function"] = "not included, because cannot pickle functions"
    info_dict_list.append(info_dict)

    gamma_hats.append(gamma_hat)
    gamma_std.append(info_dict["std_errors"][0])
    l_r_list.append(quantile_lr)
    w_t_lsit.append(quantile_w)
