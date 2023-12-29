import unittest

from estimation.sample_buckets.network.create_digraph_for_taste_shock import create_digraph_for_taste_shock_network
from estimation.sample_buckets.network.sample_taste_shock import sample_taste_shock_network
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from util.graph_functions.adj_matrix_array import *
from util.graph_functions.random_digraph import create_digraph_reciprocity
from util.graph_functions.s_functions import s_reciprocity


class IntegrationTestForReciprocity(unittest.TestCase):

    def test_reciprocity(self):
        s_function = s_reciprocity

        # 1) parse the graph and node_dict into the matrix input shape for the logit model
        for gamma in [0, 0.1, 0.5, 1]:
            n = 40
            params = np.array([0])
            erdoes_reny_model = ErdoesRenyModel(n)

            gamma = gamma
            adj_m = create_digraph_reciprocity(n, params, erdoes_reny_model, gamma)
            taste_shock, importance_sample_weight = sample_taste_shock_network(gamma, params, erdoes_reny_model,
                                                                               s_function, adj_m)
            adj_m_recreated = create_digraph_for_taste_shock_network(taste_shock, params, erdoes_reny_model, gamma,
                                                                     s_function)
            np.testing.assert_almost_equal(adj_m_recreated, adj_m, decimal=7)
