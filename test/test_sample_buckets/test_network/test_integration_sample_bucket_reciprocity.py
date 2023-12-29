import unittest

import numpy as np

from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from estimation.sample_buckets.buckets.sample_from_bucket import sample_from_bucket
from estimation.sample_buckets.network.create_digraph_for_taste_shock import create_digraph_for_taste_shock_network
from estimation.sample_buckets.network.from_taste_shock_to_bucket import from_shock_to_bucket_considering_noLink
from estimation.sample_buckets.network.sample_taste_shock import sample_taste_shock_network
from util.graph_functions.adj_matrix_array import matrix_to_array, array_to_adj_matrix
from util.graph_functions.random_digraph import create_digraph_reciprocity
from util.graph_functions.s_functions import s_reciprocity


class IntegrationTestForReciprocity(unittest.TestCase):

    def test_reciprocity(self):
        s_function = s_reciprocity

        # 1) parse the graph and node_dict into the matrix input shape for the logit model
        for gamma in [0.1, 0.5, 1]:
            n = 40
            gamma = gamma

            params = np.array([-1])
            erdoes_reny_model = ErdoesRenyModel(n)

            adj_m = create_digraph_reciprocity(n, params, erdoes_reny_model, gamma)
            taste_shock, importance_sample_weight = sample_taste_shock_network(gamma, params, erdoes_reny_model,
                                                                               s_function, adj_m)
            taste_shock_1_flatten = matrix_to_array(taste_shock)

            bucket = from_shock_to_bucket_considering_noLink(gamma, erdoes_reny_model, params,
                                                             taste_shock_1_flatten, adj_m, s_reciprocity)

            taste_shock_2_flatten = sample_from_bucket(bucket, erdoes_reny_model, params, gamma)
            taste_shock_2 = array_to_adj_matrix(taste_shock_2_flatten)
            adj_m_recreated = create_digraph_for_taste_shock_network(taste_shock_2, params, erdoes_reny_model, gamma,
                                                                     s_function)
            np.testing.assert_almost_equal(adj_m_recreated, adj_m, decimal=7)
