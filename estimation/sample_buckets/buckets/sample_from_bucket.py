import random

def sample_from_bucket(bucket, model, params,  gamma):

    threshold_at_gamma_0 = model.forward(params)
    return random.uniform(threshold_at_gamma_0+ bucket.bucket_starts*gamma,threshold_at_gamma_0+ bucket.bucket_ends*gamma)
