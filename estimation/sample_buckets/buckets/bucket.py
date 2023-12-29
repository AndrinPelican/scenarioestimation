"""
See paper: Bucket estimation

There are MN desition over the whole game

for each desition there is a bucket with start and end value


The bucket is assumed to be with

mue = 0 , mue is a vector
and
gamma = 1

For a parametrisation we get the right bucket values like:
mue + gamma * bucket

"""

class Bucket:

    def __init__(self, MN, bucket_starts, bucket_ends ):
        self.MN = MN
        self.bucket_starts = bucket_starts # -10000000, means -\infty
        self.bucket_ends = bucket_ends # 1000000 means \infty





