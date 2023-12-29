

import unittest

from util.graph_functions.adj_matrix_array import *
from util.graph_functions.s_functions import s_support


class TestSFunction(unittest.TestCase):

    def test_random_matrix(self):
        # 1) parse the graph and node_dict into the matrix input shape for the logit model

        adj_m = np.array(
            [
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0]
            ]
        )

        s_array = s_support(adj_m)

        np.testing.assert_equal(s_array[1,1],0)
        np.testing.assert_equal(s_array[1,2],2)
        np.testing.assert_equal(s_array[2,1],2)
        np.testing.assert_equal(s_array[3,3],0)
        np.testing.assert_equal(s_array[1,3],0)
        print(s_array)