import unittest

from estimation.estimation_util.make_target_function_to_model import make_target_function_for_model
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from test.mock.mock_buckets import *


class TestSimpleDensityModel(unittest.TestCase):

    def test_simple_density_model(self):
        # 1) parse the graph and node_dict into the matrix input shape for the logit model

        bucket = mock6erBuckerLeftestMiddleRightest()
        buckets = [bucket]
        bucket_weights_tilde = [1]
        density_model = ErdoesRenyModel(3)

        log_value_and_score_stable, log_value_and_score, value_and_score = make_target_function_for_model(buckets,
                                                                                                          bucket_weights_tilde,
                                                                                                          density_model)

        params = np.array([1, 0])
        cost, derivative = value_and_score(params)

        self.assertAlmostEqual(derivative[1], 0, places=4)
        self.assertTrue(derivative[
                            0] < 0)  # derivative must be negative (making gamma bigger increases the probability, decreases cost)
