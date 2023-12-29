

import unittest

from estimation.estimators.peer_effects import detect_component_lists
from test.mock.mock_buckets import *

adj_m = np.array([
                 [0, 1, 0,1,0],
                 [0, 0, 0,0,0],
                 [0, 0, 0,0,0],
                 [1, 0, 0,0,0],
                 [0, 0, 1,0,0]
             ])

comp_1 = [0, 1, 3]
comp_2 = [2, 5]

class test_split_up_peer_network(unittest.TestCase):



    """
    This network consists of 2 components
    """




    def test_split_up_peer_network_to_list(self):
        # 1) parse the graph and node_dict into the matrix input shape for the logit model

        components = detect_component_lists(adj_m)
        self.assertEqual(components, [comp_1, comp_2]) # derivative must be negative (making gamma bigger increases the probability, decreases cost)


    def test_split_up_peer_network(self):
        idexes = [True, True,  False, True, False]
        adj_m_small =adj_m[:,idexes][idexes]
        np.testing.assert_almost_equal(adj_m_small,np.array([
                 [0, 1, 1],
                 [0, 0, 0],
                 [1, 0, 0],
             ]))
















