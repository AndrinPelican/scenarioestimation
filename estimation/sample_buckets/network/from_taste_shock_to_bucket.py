import numpy as np

from estimation.sample_buckets.buckets.bucket import Bucket
from util.graph_functions.adj_matrix_array import matrix_to_array


def from_shock_to_bucket_considering_noLink(gamma, model, params, taste_shock, adj_m, s_function):
    # preparation
    thresholds_gamma_0_m = model.forward(params)
    s_values_of_graph = s_function(adj_m)
    s_values_flatten = matrix_to_array(s_values_of_graph)

    bucket_starts = []
    bucket_ends = []
    MN = taste_shock.shape[0]

    # normalize taste shock
    taste_shock_normalized = (taste_shock - thresholds_gamma_0_m) / gamma
    adj_m_flatten = matrix_to_array(adj_m)

    for k in range(MN):

        shock_normalized = taste_shock_normalized[k]
        current_s_value = s_values_flatten[k]

        if adj_m_flatten[k] == 1:

            # Here the partition of the shock possibilities when link is made
            # first bucket
            if shock_normalized <= 0:
                bucket_start = -10 ** 10
                bucket_end = 0
            else:
                # inner buckets
                for s_value in range(int(round(current_s_value) - 1)):
                    if s_value < shock_normalized and shock_normalized <= s_value + 1:
                        bucket_start = s_value
                        bucket_end = s_value + 1

                # The last bucket:
                if current_s_value - 1 < shock_normalized and shock_normalized <= current_s_value:
                    bucket_start = current_s_value - 1
                    bucket_end = current_s_value
                if shock_normalized > current_s_value:  # + 0.01:

                    print(
                        " this error should never happen, it is probably due to numerical rounding error and then tresholding the shock")
                    assert False
        else:
            # the s value of the graph has been calculated with gamma = 1 see above
            bucket_start = s_values_flatten[k]
            bucket_end = 10 ** 10

        bucket_starts.append(bucket_start)
        bucket_ends.append(bucket_end)

    bucket = Bucket(MN, np.array(bucket_starts), np.array(bucket_ends))
    return bucket
