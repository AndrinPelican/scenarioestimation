import unittest

from util.graph_functions.adj_matrix_array import *


class TestAdjMatrixArray(unittest.TestCase):

    def test_random_matrix(self):
        # 1) parse the graph and node_dict into the matrix input shape for the logit model

        adj_m = np.random.rand(74,74)

        array = matrix_to_array(adj_m)
        adj_m_2 = array_to_adj_matrix(array)
        array2 = matrix_to_array(adj_m_2)


        np.testing.assert_almost_equal(array, array2, decimal=8)
        self.assertAlmostEqual(adj_m_2[1,7], adj_m[1,7], places=8)

