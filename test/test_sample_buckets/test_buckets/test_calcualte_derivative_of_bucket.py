import unittest

import numpy as np
from scipy.stats import norm

from estimation.sample_buckets.buckets.calculate_probability_of_bucket import calculate_derivative, \
    calculate_probabilities_for_each_decision, calculate_density_at_start, calculate_density_at_end, \
    calculate_derivative_gamma_decision_wise
from test.mock.mock_buckets import mock3erBuckerLeftestMiddleRightest, mock2erBuckerLeftestRightest


class TestDerivative(unittest.TestCase):

    def test_derivative2(self):
        # 1) parse the graph and node_dict into the matrix input shape for the logit model

        bucket = mock2erBuckerLeftestRightest()
        derivative_mu, derivative_gamma = calculate_derivative(bucket, np.array([0, 0]), 1)

        self.assertAlmostEqual(0, derivative_gamma, places=9)
        self.assertAlmostEqual(derivative_mu[0], -derivative_mu[1], places=9)

    def test_derivative_gamma_decision_wise(self):
        bucket = mock3erBuckerLeftestMiddleRightest()
        mue = np.array([0, 0, 0])
        gamma = 1

        # preparation, like in calculate_derivative
        probabilities_for_each_decision = calculate_probabilities_for_each_decision(bucket, mue, gamma)
        density_at_start = calculate_density_at_start(bucket, mue, gamma)
        density_at_end = calculate_density_at_end(bucket, mue, gamma)
        derivative_start = density_at_start / probabilities_for_each_decision
        derivative_end = density_at_end / probabilities_for_each_decision

        derivative_gamma_decision_wise = calculate_derivative_gamma_decision_wise(bucket, derivative_start,
                                                                                  derivative_end)

        self.assertAlmostEqual(0, derivative_gamma_decision_wise[0], places=9)
        self.assertAlmostEqual(norm.pdf(1.96) * 1.96 * 2 / 0.95, derivative_gamma_decision_wise[1], places=5)
        self.assertAlmostEqual(0, derivative_gamma_decision_wise[2], places=9)
