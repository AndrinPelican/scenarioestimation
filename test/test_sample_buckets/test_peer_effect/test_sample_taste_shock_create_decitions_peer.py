import unittest

from estimation.sample_buckets.buckets.sample_from_bucket import sample_from_bucket
from estimation.sample_buckets.peer_effects.bucket_from_shook_peereffect import bucket_from_shook_peer_effect
from estimation.sample_buckets.peer_effects.sample_taste_shock_peer_effect import sample_taste_shock_peer_effects
from estimation.model.model_linear import LinearModel
from monte_carlo_experiment.peer_effects.create_frendschip_network.create_5_neighbours import peer_network_5_neighbours
from monte_carlo_experiment.peer_effects.create_frendschip_network.sampling_peer_decition_plain import \
    derive_desicions_from_taste_shock_peer_effects, sampling_peer_decisions
from util.graph_functions.adj_matrix_array import *

N = 120


class IntegrationTestForPeer(unittest.TestCase):

    def test_peer(self):

        # 1) parse the graph and node_dict into the matrix input shape for the logit model
        for gamma in [0, 0.1, 0.5, 1]:
            a = np.array([0])

            adj_m_peers = peer_network_5_neighbours(n_agents=N)
            X = np.ones((adj_m_peers.shape[0], 1))
            model = LinearModel(X)

            y_0, _ = sampling_peer_decisions(params=a, gamma=gamma, adj_m_peers=adj_m_peers, X=X)
            taste_shock, log_importance_sampling_weight = sample_taste_shock_peer_effects(y=y_0, peer_adj_m=adj_m_peers,
                                                                                          gamma_for_sampling=gamma,
                                                                                          params=a, model=model)
            thresholds_gamma_0 = model.forward(a)
            y_1, _ = derive_desicions_from_taste_shock_peer_effects(N, gamma, thresholds_gamma_0, adj_m_peers,
                                                                    taste_shock)
            np.testing.assert_almost_equal(y_1, y_0, decimal=7)

    def test_peer_with_buckets(self):
        # 1) parse the graph and node_dict into the matrix input shape for the logit model

        for gamma in [0.1, 0.5, 1]:
            params = np.array([0])
            adj_m_peers = peer_network_5_neighbours(n_agents=N)
            X = np.ones((adj_m_peers.shape[0], 1))
            model = LinearModel(X)

            thresholds_gamma_0 = model.forward(params)

            y_0, _ = sampling_peer_decisions(params=params, gamma=gamma, adj_m_peers=adj_m_peers, X=X)

            taste_shock, log_importance_sampling_weight = sample_taste_shock_peer_effects(y=y_0, peer_adj_m=adj_m_peers,
                                                                                          gamma_for_sampling=gamma,
                                                                                          params=params, model=model)
            bucket = bucket_from_shook_peer_effect(y=y_0, peer_adj_m=adj_m_peers, taste_shock=taste_shock,
                                                   gamma_for_sampling=gamma, params=params, model=model)

            taste_shock_2 = sample_from_bucket(bucket, model, params, gamma)

            y_1, _ = derive_desicions_from_taste_shock_peer_effects(N, gamma, thresholds_gamma_0, adj_m_peers,
                                                                    taste_shock_2)
            np.testing.assert_almost_equal(y_1, y_0, decimal=7)
