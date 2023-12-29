import numpy as np

from estimation.sample_buckets.buckets.bucket import Bucket


def mock3erBuckerLeftestMiddleRightest():
    """
    Bucket:
    [-1000  , 0]
    [-1.96  , 1.96]
    [0      , 1000]
    """

    bucket_starts = np.array([-1000, -1.96, 0])
    bucket_ends = np.array([0, 1.96, 100000])
    bucket = Bucket(3, bucket_starts=bucket_starts, bucket_ends=bucket_ends)

    return bucket


def mock6erBuckerLeftestMiddleRightest():
    """
    Bucket:
    [-1000  , 0]
    [-1.96  , 1.96]
    [0      , 1000]
     [-1000  , 0]
    [-1.96  , 1.96]
    [0      , 1000]
    """

    bucket_starts = np.array([-1000, -1.96, 0, -1000, -1.96, 0])
    bucket_ends = np.array([0, 1.96, 100000, 0, 1.96, 100000])
    bucket = Bucket(6, bucket_starts=bucket_starts, bucket_ends=bucket_ends)

    return bucket


def mock6erBuckerFromMinus1To1():
    """
    Bucket:
    [-1000  ,-1]
    [-1  , 1]
    [1     , 1000]
    [-1000  ,-1]
    [-1  ,  1]
    [1     , 1000]
    """

    bucket_starts = np.array([-1000, -1, 1, -1000, -1, 1])
    bucket_ends = np.array([-1, 1, 100000, -1, 1, 100000])
    bucket = Bucket(6, bucket_starts=bucket_starts, bucket_ends=bucket_ends)

    return bucket


def mock2erBuckerLeftestRightest():
    """
    Bucket:
    [-1000  , 0]
    [0      , 1000]
    """

    bucket_starts = np.array([-1000, 0])
    bucket_ends = np.array([0, 1000])
    bucket = Bucket(2, bucket_starts=bucket_starts, bucket_ends=bucket_ends)

    return bucket


def mock4erBucketMarkedEntry():
    """
    Bucket:
    [ -1   ,1000]
    [- 1000 , -2]
    [- 2    , -1]
    [- 2    , -1]
    """

    bucket_starts = np.array([-1, -1000, -2, -2])
    bucket_ends = np.array([1000, -2, -1, -1])
    bucket = Bucket(4, bucket_starts=bucket_starts, bucket_ends=bucket_ends)

    return bucket
