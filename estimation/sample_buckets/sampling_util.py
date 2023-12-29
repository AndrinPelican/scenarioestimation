import random
from scipy.stats import norm



def sample_from_above(treshold):
    # sample out of partial distribution
    # y is the name of the axis for the cdf
    y = norm.cdf(treshold)
    sample_on_y = random.uniform(y,1)
    sample = norm.ppf(sample_on_y) # ppf Percent point function (inverse of cdf — percentiles).

    # here maybe numerical dificutiey, which can lead to downs tream error due to trasholding
    assert sample>treshold

    return sample, 1-y

def sample_from_below(treshold):
    # sample out of partial distribution
    # y is the name of the axis for the cdf
    y = norm.cdf(treshold)
    sample_on_y = random.uniform(0,y)
    sample = norm.ppf(sample_on_y)  # ppf Percent point function (inverse of cdf — percentiles).

    # here maybe numerical dificutiey, which can lead to downs tream error due to trasholding
    assert sample < treshold

    return sample , y



