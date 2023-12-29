import unittest
import numpy as np

from monte_carlo_experiment.peer_effects.create_frendschip_network.sampling_peer_decition_plain import \
    derive_desicions_from_taste_shock_peer_effects


class TestFromShockToDecitionRealisation(unittest.TestCase):

    def test_market_entry(self):

        adj_m = np.array([
            [0,1,1],
            [0,0,1],
            [0,0,0]
        ])
        tresholds_gamma_0 = np.array([0,0,0])

        # Test different cases

        taste_shock = np.array([0.5, 0.5, 0.5])
        y, added_due_to_interaction = derive_desicions_from_taste_shock_peer_effects(n=3, gamma=1, thresholds_gamma_0=tresholds_gamma_0, adj_m_peers=adj_m, taste_shocks=taste_shock)
        np.testing.assert_almost_equal(np.array([0,0,0]), y)

        taste_shock = np.array([-0.1, 0.5, 0.5])
        y, added_due_to_interaction = derive_desicions_from_taste_shock_peer_effects(n=3, gamma=1, thresholds_gamma_0=tresholds_gamma_0, adj_m_peers=adj_m,
                                                           taste_shocks=taste_shock)
        np.testing.assert_almost_equal(np.array([1, 0, 0]), y)

        taste_shock = np.array([0.1, -0.5, 0.5])
        y, added_due_to_interaction = derive_desicions_from_taste_shock_peer_effects(n=3, gamma=1, thresholds_gamma_0=tresholds_gamma_0, adj_m_peers=adj_m,
                                                           taste_shocks=taste_shock)
        np.testing.assert_almost_equal(np.array([1, 1, 0]), y)

        taste_shock = np.array([0.1, 0.5, -0.5])
        y, added_due_to_interaction = derive_desicions_from_taste_shock_peer_effects(n=3, gamma=1, thresholds_gamma_0=tresholds_gamma_0, adj_m_peers=adj_m,
                                                           taste_shocks=taste_shock)
        np.testing.assert_almost_equal(np.array([1, 1, 1]), y)






