import numpy as np


def print_q_statistic(log_important_sampling_weights):
    if log_important_sampling_weights == None:
        return None

    sum_of_weights_divided_by_max = 0
    log_of_max = max(log_important_sampling_weights)
    for log_weight in log_important_sampling_weights:
        weight_divided_by_max = np.exp(log_weight - log_of_max)
        sum_of_weights_divided_by_max += weight_divided_by_max
    print("Q statistic (max_weight/sumOfWeights): " + str(
        1 / sum_of_weights_divided_by_max) + "     See \"The sample size required in importance sampling\"")
    print_effective_sample_size(log_important_sampling_weights)


def print_effective_sample_size(important_sampling_weights):
    if important_sampling_weights == None:
        return None

    sum_of_weights = 0
    squared_sum_of_weights = 0
    max_of_List = max(important_sampling_weights)
    for log_weight in important_sampling_weights:
        weight = np.exp(log_weight - max_of_List)
        sum_of_weights += weight
        squared_sum_of_weights += weight ** 2
    print("effective sample size: " + str(sum_of_weights ** 2 / squared_sum_of_weights))
