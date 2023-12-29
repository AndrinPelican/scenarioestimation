import math

import numpy as np

from estimation.sample_buckets.buckets.bucket import Bucket


def bucket_from_shook_peer_effect(y, peer_adj_m, taste_shock, gamma_for_sampling, params, model):
    bucket_starts = []
    bucket_ends = []
    N = taste_shock.shape[0]
    thresholds_for_y_observed = np.matmul(peer_adj_m, y)

    # while simulating we shift by thresholds_gamma_0 and scaled down by gamma, here reverse it
    # this way we can tread gamma like it is 1
    thresholds_gamma_0 = model.forward(params)
    shock_normalized = (taste_shock - thresholds_gamma_0) / gamma_for_sampling

    for k in range(N):

        shock_normalized_at_k = shock_normalized[k]

        if y[k] == 0:
            bucket_start = thresholds_for_y_observed[k]
            bucket_end = 100000
        else:
            if shock_normalized_at_k < 0:
                bucket_start = -100000
                bucket_end = 0
            else:
                bucket_start = math.floor(shock_normalized_at_k)
                bucket_end = math.ceil(shock_normalized_at_k)

        bucket_starts.append(bucket_start)
        bucket_ends.append(bucket_end)

    bucket = Bucket(N, np.array(bucket_starts), np.array(bucket_ends))
    return bucket
