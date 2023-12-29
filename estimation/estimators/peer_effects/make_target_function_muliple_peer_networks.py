
from scipy.sparse.csgraph._traversal import connected_components
from estimation.estimators.peer_effects.make_target_function_peer_effects import \
    create_buckets_make_target_function_for_model_peer, create_buckets_make_target_function_for_model_peer_independent
import numpy as np

from util.print_effective_sample_size import print_q_statistic

is_independent = False

def make_target_function_multiple_peer_networks(y, X, peer_adj_m, initial_params, gamma_for_sampling, n_buckets = 100,  consider_separate_networks =True):

    print("attention fixme is indipendent: "+str(is_independent))

    if consider_separate_networks:

        # finding the components
        n_components, labels = connected_components(peer_adj_m)
        log_value_and_score_functions = []

        for i in range(n_components):
            print("\nIn game " + str(i))

            indexes_of_components = labels == i
            X_i = X[indexes_of_components]
            y_i = y[indexes_of_components]
            peer_adj_m_i = peer_adj_m[:,indexes_of_components][indexes_of_components]
            if is_independent:
                log_value_and_score_stable, _, _, important_sampling_weights = create_buckets_make_target_function_for_model_peer_independent(y=y_i, X=X_i, peer_adj_m=peer_adj_m_i, initial_params=initial_params, gamma_for_sampling =gamma_for_sampling, n_buckets=n_buckets)
            else:
                log_value_and_score_stable, _, _, important_sampling_weights = create_buckets_make_target_function_for_model_peer(y=y_i, X=X_i, peer_adj_m=peer_adj_m_i, initial_params=initial_params, gamma_for_sampling =gamma_for_sampling, n_buckets=n_buckets)


            print_q_statistic(important_sampling_weights)

            log_value_and_score_functions.append(log_value_and_score_stable)

        def combined_value_and_score(x):
            score = 0
            value = 0
            for current_value_and_score_function in log_value_and_score_functions:
                current_value, current_score = current_value_and_score_function(x)
                score += current_score
                value += current_value
            score = score
            value = value
            if (is_independent):
                value = 0.5*np.linalg.norm(score)**2 # depending on method, you may want to minimize the square gradient
            return value, score

        return combined_value_and_score, None,None,None

    else:
        return create_buckets_make_target_function_for_model_peer(y=y, X=X, peer_adj_m=peer_adj_m, initial_params=initial_params, gamma_for_sampling =gamma_for_sampling, n_buckets=n_buckets)


def detect_component_lists(adj):
    n_components, labels = connected_components(adj)
    return labels