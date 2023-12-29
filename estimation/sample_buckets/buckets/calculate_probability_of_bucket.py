from scipy.stats import norm
import numpy as np

"""
This module is responsible to calculate bucket probability and derivatives.

It is numerically challenging because probability of areas of the normal distribution faar from 0 are very small, so 
that rounding arrows or underflow become relevant. 

For rounding error see approach in calculate_probabilities_for_each_decision. 

For Underflow, nothing done so far, but you could only work with log probabilities:
- if x is far form 0 only bucket border closer to 0 matters, the other could be neglected.
- you can approximate the cdf at the tail as 1/x * pdf, see http://www.stat.yale.edu/~pollard/Books/Mini/Basic.pdf



"""
def calculate_log_probability(bucket, mue, gamma):
    """

    :param bucket: A bucket
    :param mue: the threshold values for each decision, a n vector
    :param gamma: The strategic parameter scalar
    :return:
    """

    probabilities_for_each_decision = calculate_probabilities_for_each_decision(bucket, mue, gamma)

    log_probability = 0
    i = 0
    for value in probabilities_for_each_decision:

        if value < 10**-200:
            # The value is 0 because one bucket is placed so far outside the distibution that it is rounded to 0
            # for numerical reasons, in order to cicumvent this problem we set the value to  10^-100000, so the log
            # to -100000 this way the bucket is left out of consideration if there are still buckets with numerically
            # positive probabilities.
            print("Warning bucket in one dimension really unlikely: this leads to numerical instability")
            print(value)
            print(gamma)
            print(mue)
            print(mue[i])
            print("bucket start  " + str(bucket.bucket_starts[i]))
            print("bucket end    " + str(bucket.bucket_ends[i]))
            value_recalculated = norm.cdf(mue[i] + bucket.bucket_ends[i] * gamma) - norm.cdf(mue[i] + bucket.bucket_starts[i] * gamma)
            print("value recalculated :  "+ str(value_recalculated))
        else:
            log_probability += np.log(value)

        i += 1

    return log_probability


def calculate_probabilities_for_each_decision(bucket, mue, gamma):
    threshold_end = mue + bucket.bucket_ends * gamma
    threshold_start = mue + bucket.bucket_starts * gamma

    """
    Note: 
    for symmetry reason we have: 
        norm.cdf(threshold_end) - norm.cdf(threshold_start) =  norm.cdf(-threshold_start) - norm.cdf(-threshold_end)
    
    However numerically there is a difference: when the cdf is close to 1, the precision of the difference is about 
    10^16, (determined by the position after the comma in the floating point representation) 
    if it is close to 0 it is about 10^308, determined by the exponent in the floating point
    """

    return np.max([norm.cdf(threshold_end) - norm.cdf(threshold_start),norm.cdf(-threshold_start) - norm.cdf(-threshold_end)], axis=0)

def calculate_density_at_start(bucket, mue, gamma):
    return norm.pdf(mue + bucket.bucket_starts * gamma)

def calculate_density_at_end(bucket, mue, gamma):
    return norm.pdf(mue + bucket.bucket_ends * gamma)

def calculate_derivative_gamma_decision_wise(bucket, derivative_start, derivative_end):
    return bucket.bucket_ends * derivative_end - bucket.bucket_starts * derivative_start

def calculate_derivative(bucket, mue, gamma):
    """
    :param bucket: A bucket
    :param mue: the treshold values for each decition, a n vector
    :param gamma: The strategic parameter scalar
    :return:
    """

    probabilities_for_each_decision = calculate_probabilities_for_each_decision(bucket, mue, gamma)
    density_at_start = calculate_density_at_start(bucket, mue, gamma)
    density_at_end = calculate_density_at_end(bucket, mue, gamma)

    derivative_start = density_at_start/probabilities_for_each_decision
    derivative_end = density_at_end/probabilities_for_each_decision

    derivative_mue = derivative_end - derivative_start  # a n vector
    derivative_gamma = np.sum(calculate_derivative_gamma_decision_wise(bucket, derivative_start, derivative_end))

    return derivative_mue, derivative_gamma
