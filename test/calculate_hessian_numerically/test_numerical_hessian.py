import unittest

from scipy.optimize import rosen, rosen_der, rosen_hess

from estimation.estimation_util.calculate_hession_numerically import calculate_hessian_numerically
from test.mock.mock_buckets import *

"""
We test the numerical hessian calculation.

We use the rosenbock functions, to compare the analytical results  with our numerical.   

"""


def value_and_score(params):
    value = rosen(params)
    der = rosen_der(params)
    return value, der


class TestNumericalHessian(unittest.TestCase):

    def test_rosenbock(self):
        params = 0.1 * np.arange(4)

        numerical_hessian = calculate_hessian_numerically(value_and_score, params)
        exact_hessian = rosen_hess(params)
        np.testing.assert_almost_equal(exact_hessian, numerical_hessian, 0.002)
