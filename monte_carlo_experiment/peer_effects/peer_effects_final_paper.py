"""
This is a monte carlo experiment

"""

import pickle
import numpy as np

from estimation.estimators.peer_effects.estimate_peer_effects import estimate_model_peer_effect
from monte_carlo_experiment.monte_carlo_util.likelyhood_ratio import likelyhood_ratio
from monte_carlo_experiment.monte_carlo_util.wald_test import wald_test
from monte_carlo_experiment.peer_effects.create_frendschip_network.create_5_neighbours import block_random_geometric_graph
from monte_carlo_experiment.peer_effects.create_frendschip_network.sampling_peer_decition_plain import \
    sampling_peer_decisions

gamma = 0.2
n_szenarios_list = [25, 50, 100]
n_szenarios_list = [ 1, 10, 100]
# n_szenarios_list = [1, 2, 1]
n_simulations = 100 # should be 1000
# n_simulations = 225
n_runs = 1


name = "run_28"
comment = "testrun"


for n_szenarios in n_szenarios_list:
    for is_independent_games in [True, False]:
        if (is_independent_games):
            n_agents_total = 2000
            n_blocks = 100
        else:
            n_agents_total = 500
            n_blocks = 1


        params = np.array([-1, -0.5, -1, +0.5])
        # binary matrix for the two groups
        X = np.zeros((n_agents_total, 4))
        group_1 = np.random.randint(2, size=(n_agents_total))
        X[:, 0] = group_1
        X[:, 1] = 1 - group_1
        X[:, 2] = np.random.uniform(0, 1, size=(n_agents_total))
        X[:, 3] = np.random.uniform(0, 1, size=(n_agents_total))

        gamma_hats = []
        gamma_std = []
        denitys = []
        n_added_list = []

        """
        Different configurations on how peers are connected
        """
        adj_m_peers = block_random_geometric_graph(n_agents=n_agents_total, n_blocks=n_blocks)
        average_degree = sum(sum(adj_m_peers)) / (n_agents_total)
        density = sum(sum(adj_m_peers))
        reciprocity = sum(sum(adj_m_peers * np.transpose(adj_m_peers, [1, 0]))) / denitys

        l_r_list = []
        w_t_lsit = []
        info_dict_list = []

        for i in range(n_simulations):
            print("==================================================================" + str(i))
            y, n_added = sampling_peer_decisions(gamma=gamma, params=params, adj_m_peers=adj_m_peers, X=X, report_adding=True)
            print("Density:  " + str(sum(y) / n_agents_total))
            denitys.append(sum(y) / (n_agents_total))
            n_added_list.append(n_added)
            gamma_hat, a_hat, info_dict = estimate_model_peer_effect(y=y, X=X, peer_adj_m=adj_m_peers, initial_params=params,
                                                                     initial_gamma=gamma, n_buckets=n_szenarios, n_runs=n_runs,
                                                                     consider_separate_networks=True)
            l_r, quantile_lr = likelyhood_ratio(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
            w_t, quantile_w = wald_test(params_null=np.array([gamma] + list(params)), info_dict=info_dict)
            info_dict["l_r"] = quantile_lr
            info_dict["quantile_lr"] = quantile_lr
            info_dict["quantile_w"] = quantile_w
            info_dict["w_t"] = quantile_w
            info_dict["density_of_y"] = sum(y) / (n_agents_total)
            info_dict["y"] = y
            info_dict["log_likelihood_value_and_score_function"] = "not included, because cannot pickle functions"
            info_dict_list.append(info_dict)

            gamma_hats.append(gamma_hat)
            gamma_std.append(info_dict["std_errors"][0])
            l_r_list.append(quantile_lr)
            w_t_lsit.append(quantile_w)

        pickle_dict = {
            "info_dict_list": info_dict_list,
            "n_agents_total": n_agents_total,
            "params": params,
            "gamma": gamma,
            "n_szenarios": n_szenarios,
            "n_runs": n_runs,
            "name_network_creation": block_random_geometric_graph.__name__,
            "n_blocks": n_blocks,
            "block_size": n_agents_total / n_blocks,
            "average_degree": average_degree,
            "density": density,
            "reciprocity": reciprocity,
            "adj_m": adj_m_peers,
            "X": X,
            "comment": comment
        }
        file_to_save = open("./saved_simulations/" + name+"__"+str(int(n_agents_total/n_blocks)) +"_player_per_game__"+str(n_blocks)+"_indeptendnt_games__"+str(n_szenarios)+"_szenarios.pickle" , 'wb')
        pickle.dump(pickle_dict, file_to_save)

