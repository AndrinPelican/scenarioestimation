
import numpy as np
from util.graph_functions.adj_matrix_array import array_to_adj_matrix


def check_bucket_reciprocity(bucket, ajd_m):

    bucket_array = np.array([bucket.bucket_starts, bucket.bucket_ends])

    is_inner =np.logical_and( -1< bucket_array[0,:] , bucket_array[1,:]< 2)

    is_inner = array_to_adj_matrix(is_inner)

    # check that only one inner of reciprocical links is inner
    assert False == np.logical_and(np.transpose(is_inner, [1,0]), is_inner).any()

    # check that
    allo_with_one_in_inner = np.logical_or(np.transpose(is_inner, [1, 0]), is_inner)
    assert  False == np.logical_and(allo_with_one_in_inner, np.logical_not(ajd_m)).any()

    print("depug, take me out")