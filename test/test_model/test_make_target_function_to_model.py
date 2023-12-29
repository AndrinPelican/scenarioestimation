import unittest

from estimation.estimation_util.make_target_function_to_model import make_target_function_for_model
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from test.mock.mock_buckets import *


class TestIfStableTargetFunctionGivesSameValuesAsNormal(unittest.TestCase):

    def test_stable_vs_normal(self):
        # 1) parse the graph and node_dict into the matrix input shape for the logit model

        bucket = mock6erBuckerLeftestMiddleRightest()
        buckets = [bucket, bucket, bucket]
        bucket_weights_tilde = [1, 1, 1]
        density_model = ErdoesRenyModel(3)

        log_value_and_score_stable, log_value_and_score, target_value_and_score = make_target_function_for_model(
            buckets, bucket_weights_tilde, density_model)

        params = np.array([1, 0])

        cost, derivative = log_value_and_score(params)
        cost_stable, derivative_stable = log_value_and_score_stable(params)

        np.testing.assert_almost_equal(cost, cost_stable, decimal=5)
        np.testing.assert_almost_equal(derivative, derivative_stable, decimal=5)
