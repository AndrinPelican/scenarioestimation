import unittest

from estimation.estimation_util.make_target_function_to_model import make_target_function_for_model
from estimation.estimation_util.optimizer import estimate_parameters
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from test.mock.mock_buckets import *


class TestEstimateParamsForBuckets(unittest.TestCase):

    def test_mock6erBuckerLeftestMiddleRightest(self):
        # 1) parse the graph and node_dict into the matrix input shape for the logit model

        bucket = mock6erBuckerFromMinus1To1()
        buckets = [bucket]
        bucket_weights_tilde = [1]
        density_model = ErdoesRenyModel(3)
        log_value_and_score_stable, log_value_and_score, value_and_score = make_target_function_for_model(buckets,
                                                                                                          bucket_weights_tilde,
                                                                                                          density_model)

        x0 = density_model.get_initial_params()
        optimal_params, optimal_target_value = estimate_parameters(value_and_score, x0)

        # you can see that it must be 0 for symetry reasons
        self.assertAlmostEqual(optimal_params[1], 0, places=2)
        # see whether you can calcualte by hand the correct result
        self.assertTrue(optimal_params[0] > 0.1)

        # see invariant for new starting value
        x0 = np.array([0.5, 0.5])
        optimal_params_2, optimal_target_value = estimate_parameters(value_and_score, x0)
        self.assertAlmostEqual(optimal_params_2[0], optimal_params[0], places=2)
        self.assertAlmostEqual(optimal_params_2[1], optimal_params[1], places=2)

        # see invariant when you enter the same bucket several times
        buckets = [bucket, bucket, bucket]
        bucket_weights_tilde = [1, 0.5, 0.5]
        log_value_and_score_stable, log_value_and_score, value_and_score = make_target_function_for_model(buckets,
                                                                                                          bucket_weights_tilde,
                                                                                                          density_model)
        x0 = density_model.get_initial_params()
        optimal_params3, optimal_target_value = estimate_parameters(value_and_score, x0)

        self.assertAlmostEqual(optimal_params3[0], optimal_params[0], places=2)
        self.assertAlmostEqual(optimal_params3[1], optimal_params[1], places=2)
