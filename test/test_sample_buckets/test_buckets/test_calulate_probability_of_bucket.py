import unittest

import numpy as np

from estimation.sample_buckets.buckets.calculate_probability_of_bucket import calculate_log_probability
from test.mock.mock_buckets import mock3erBuckerLeftestMiddleRightest


class TestStringMethods(unittest.TestCase):

    def test_calculateProbability(self):
        # 1) parse the graph and node_dict into the matrix input shape for the logit model

        bucket = mock3erBuckerLeftestMiddleRightest()

        probability = calculate_log_probability(bucket, np.array([0, 0, 0]), 1)
        self.assertAlmostEqual(0.5 * 0.5 * 0.95, np.exp(probability), places=4)
