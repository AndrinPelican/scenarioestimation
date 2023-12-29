import unittest

from estimation.estimators.network.estimate_model_network import estimate_model_network_one_run
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from estimation.sample_buckets.buckets.calculate_probability_of_bucket import calculate_log_probability
from estimation.sample_buckets.network.from_taste_shock_to_bucket import from_shock_to_bucket_considering_noLink
from estimation.sample_buckets.network.sample_taste_shock import sample_taste_shock_network
from monte_carlo_experiment.reciprocity_network_exact_vs_mc.estimate_model_reciprocity_exact import \
    estimate_model_reciprocity_exact
from test.mock.mock_buckets import *
from util.graph_functions.adj_matrix_array import matrix_to_array
from util.graph_functions.s_functions import s_reciprocity


class TestEstimationForN3(unittest.TestCase):
    """
           We test whether the estimation works for simple case:
           [0,1,0],
           [1,0,0],
           [0,0,0]

           # F is the cdf of normal and f the density

           The likelyhood is:

           target =  - (1-F(a))**2 *\   # recipcoe no edgte
               (1-F(a+gamma))*F(a)\     # exge in one direction
               * (F(a) ** 2 + 2 *  (F(a + gamma)-F(a)) *F(a)) # reciproce edge


           gamma = 0.46315166
           a = 0.19511973]

           :return:
    """

    def test_buckets(self):

        adj_m = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]
        ])

        gamma_for_sampling = 0.5
        params = np.array([0])
        model = ErdoesRenyModel(3)
        s_function = s_reciprocity

        buckets = []
        bucket_weights_tilde = []

        n_type_1 = 0
        prop_type_1 = None
        n_type_2 = 0
        prop_type_2 = None
        n_type_3 = 0
        prop_type_3 = None

        anz_sim = 10000
        for _ in range(anz_sim):

            taste_shock_1, importance_sampling_log_weight = sample_taste_shock_network(gamma_for_sampling, params,
                                                                                       model, s_function, adj_m)
            taste_shock_1_flatten = matrix_to_array(taste_shock_1)
            bucket = from_shock_to_bucket_considering_noLink(gamma_for_sampling, model, params,
                                                             taste_shock_1_flatten, adj_m, s_reciprocity)

            log_p_base = calculate_log_probability(bucket, 0, gamma_for_sampling)
            log_p_tilde = log_p_base - importance_sampling_log_weight
            """
            here there should occure 3 kind of buckets
            
                             type 1             type 2             type 3
            edge_recip    [0,        1]  --- [-1000,    0]  --- [-1000,    0]
            no_edge_1_dir [1,     1000]      [1,     1000]      [1,     1000]
            edge_recip    [-1000,    0]  --- [0,        1]  --- [-1000,    0]
            no_edge       [0,   , 1000]      [0,   , 1000]      [0,   , 1000]
            edge_1_dir    [-1000,    0]      [-1000,    0]      [-1000,    0]
            no_edge       [0,   , 1000]      [0,   , 1000]      [0,   , 1000]
            """

            assert bucket.bucket_starts[1] == 1
            assert bucket.bucket_starts[5] == 0
            assert bucket.bucket_ends[5] == 10 ** 10

            if (bucket.bucket_starts[0] == 0):
                n_type_1 += 1
                prop_type_1 = np.exp(log_p_tilde)
            elif (bucket.bucket_starts[2] == 0):
                n_type_2 += 1
                prop_type_2 = np.exp(log_p_tilde)
            else:
                n_type_3 += 1
                prop_type_3 = np.exp(log_p_tilde)

            buckets.append(bucket)
            bucket_weights_tilde.append(calculate_log_probability(bucket, 0, gamma_for_sampling))

        """
        Here we assert that p_tilde -> the probability of a bucket being drawn (which differs form the bucket probability 
        to the parameters) is close to the frequency of the buckets drawn
        """

        print("number Of Type 1:   " + str(n_type_1 / anz_sim))
        print("prob Of Type 1:     " + str(prop_type_1))
        np.testing.assert_almost_equal(n_type_1 / anz_sim, prop_type_1, decimal=2)

        print("number Of Type 2:   " + str(n_type_2 / anz_sim))
        print("prob Of Type 2:     " + str(prop_type_2))
        np.testing.assert_almost_equal(n_type_2 / anz_sim, prop_type_2, decimal=2)

        print("number Of Type 3:   " + str(n_type_3 / anz_sim))
        print("prob Of Type 3:     " + str(prop_type_3))
        np.testing.assert_almost_equal(n_type_3 / anz_sim, prop_type_3, decimal=2)

    def test_1_reciprocity_link(self):

        adj_m = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]
        ])
        model = ErdoesRenyModel(3)
        params = np.array([0])

        gamma_hat, a_hat, _, _, _ = estimate_model_network_one_run(adj_m,
                                                                   model,
                                                                   s_function=s_reciprocity,
                                                                   gamma_for_sampling=0.6,
                                                                   initial_params=params,
                                                                   n_buckets=100)

        print("estimated:   " + str(gamma_hat))
        print("actual:      " + str(0.46315166))
        np.testing.assert_almost_equal(gamma_hat, 0.46315166, decimal=1)

        print("estimated:   " + str(a_hat[0]))
        print("actual:      " + str(-0.19511973))
        np.testing.assert_almost_equal(a_hat[0], -0.19511973, decimal=1)

    def test_1_reciprocity_link_exact(self):

        adj_m = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]
        ])
        model = ErdoesRenyModel(3)

        gamma_hat, a_hat, info_dict = estimate_model_reciprocity_exact(adj_m=adj_m, model=model)
        print("estimated:   " + str(gamma_hat))
        print("actual:      " + str(0.46315166))
        np.testing.assert_almost_equal(gamma_hat, 0.46315166, decimal=3)

        print("estimated:   " + str(a_hat[0]))
        print("actual:      " + str(-0.19511973))
        np.testing.assert_almost_equal(a_hat[0], -0.19511973, decimal=3)
