import numpy as np

from estimation.sample_buckets.buckets.calculate_probability_of_bucket import calculate_log_probability, \
    calculate_derivative

"""
This function takes the buckets, and the weights of the buckets and creates the target function.

"""


def make_target_function_for_model(buckets, bucket_log_weights_tilde, model):
    def log_value_and_score_stable(x):
        """
        :param x: A np.array with the parameters to optimize, it contains as first value gamma,
        :return:
        """
        gamma = x[0]
        params = x[1:]
        mue = model.forward(params)
        n_buckets = buckets.__len__()
        total_derivative = np.zeros(x.shape)

        # gamma
        if (gamma <= 0.00001):
            total_derivative = np.zeros(x.shape)
            total_derivative[0] = -100000
            return 10 ^ 30000000, total_derivative

        # fist part calculate the probailities in logs, then in weighted adding, you can ajust by the biggest probapbilty
        estimated_log_probabilities = []
        for i in range(n_buckets):
            bucket = buckets[i]
            log_probability = calculate_log_probability(bucket, mue, gamma)
            # weight adjustment
            log_w_tilde = bucket_log_weights_tilde[i]
            estimated_log_probability = log_probability - log_w_tilde
            estimated_log_probabilities.append(estimated_log_probability)

        # this is the bucket with
        max_log_probability = max(estimated_log_probabilities)

        total_probability = 0
        for i in range(n_buckets):
            # actually total_probability += np.exp(estimated_log_probabilities[i])/np.exp(max_log_probability),
            # but unstable
            # The / n_buckets is important for calculation of the right scaled hessian
            total_probability += np.exp(estimated_log_probabilities[i] - max_log_probability) / n_buckets

        # actually total_log_probability = np.log(total_probability*np.exp(max_log_probability)), but unstable
        total_log_probabilty = np.log(total_probability) + max_log_probability

        for i in range(n_buckets):
            bucket = buckets[i]
            derivative_mue, derivative_gamma = calculate_derivative(bucket, mue, gamma)
            derivative_a = model.backward(derivative_mue)
            derivative = np.insert(derivative_a, 0,
                                   derivative_gamma)  # put derivative Gamma at place 0 of derivative (push)

            # weight adjustment
            # current_probaiblity_scaled = 0
            # numerrically unstable: total_derivative += derivative np.exp(estimated_log_probabilities[i]) /np.exp(total_log_probabilty)/ n_buckets
            # The / n_buckets is important for calculation of the right scaled hessian
            total_derivative += derivative * np.exp(estimated_log_probabilities[i] - total_log_probabilty) / n_buckets

        # print("            params gamma:  " + str(gamma))
        # print("            der gamma:     " + str(total_derivative[0]))
        # print("                                            params a :        " + str(a))
        # print("                                            der derivative_a: " + str(total_derivative[1:]))
        # print(total_log_probabilty)
        return - total_log_probabilty, - total_derivative

    """
    These functions below are numerically unstable
    The small probability lead to floating point underflow, However they are simpler to understand and bridge the 
    connection to the paper.
    """

    def value_and_score(params):
        """
        :param params: A np.array with the parameters to optimize, it contains as first value gamma,
        :return:
        """
        gamma = params[0]
        a = params[1:]
        mue = model.forward(a)

        total_derivative = np.zeros(params.shape)
        total_probability = 0

        # gamma
        if (gamma <= 0.00001):
            total_derivative[0] = -10000000000000
            return total_probability, total_derivative

        n_buckets = buckets.__len__()
        for i in range(n_buckets):
            bucket = buckets[i]
            probability = np.exp(calculate_log_probability(bucket, mue, gamma))
            derivative_mue, derivative_gamma = calculate_derivative(bucket, mue, gamma)
            derivative_a = model.backward(derivative_mue)

            # weight adjustment
            w_tilde = np.exp(bucket_log_weights_tilde[i])
            estimated_probability = probability / w_tilde
            derivative = np.insert(derivative_a, 0,
                                   derivative_gamma)  # put derivative Gamma at place 0 of derivative (push)
            total_derivative += estimated_probability * derivative / n_buckets
            total_probability += estimated_probability / n_buckets

        return - total_probability, - total_derivative

    def log_value_and_score(params):
        """
            d log(f(x))/dx = 1/f(x) * d f(x) / dx
        """

        neg_prob, neg_score = value_and_score(params)

        if (neg_prob == 0):
            # neg_prob == 0 is when gammy <= 0 This case is not allowed
            return 10 ^ 300, neg_score

        log_target = -np.log(-neg_prob)
        log_score = neg_score / (-neg_prob)
        return log_target, log_score

    return log_value_and_score_stable, log_value_and_score, value_and_score
