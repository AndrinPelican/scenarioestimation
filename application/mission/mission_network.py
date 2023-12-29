from application.mission.diplomatic_mission_read_in_network import adj_m_correlates_of_war
from estimation.estimators.network.estimate_model_network import estimate_model_network
from estimation.sample_buckets.network.from_taste_shock_to_bucket import \
    from_shock_to_bucket_transitivity_considering_noLink
from estimation.model.model_linear import LinearModel
from util.graph_functions.s_functions import s_transitivity
from util.params_and_X_for_degree_heterogeneity import X_matrix_fixed_effect_for_network_n_agents

# This considers degree heterogeneity
X = X_matrix_fixed_effect_for_network_n_agents(adj_m_correlates_of_war.shape[0])
linearModel = LinearModel(X)

# This considers only two group
# X = proximity_and_two_groups(n)
# linearModel = LinearModel(X)


a_hats = []
gamma_hats = []
denitys = []

gamma_hat, a_hat, info_dict = estimate_model_network(adj_m=adj_m_correlates_of_war,
                                                     model=linearModel,
                                                     from_shock_to_bucket_monotone_increasing=from_shock_to_bucket_transitivity_considering_noLink,
                                                     s_function=s_transitivity,
                                                     n_buckets=1,
                                                     n_runs=1)
