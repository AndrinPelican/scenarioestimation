import numpy as np

from estimation.model.model_linear import LinearModel


def sampling_peer_decisions(gamma, params, adj_m_peers, X  ,report_adding = False):

    n = adj_m_peers.shape[0]
    taste_shocks = np.random.normal(loc=0.0, scale=1.0, size=(n))
    model = LinearModel(X)
    thresholds_gamma_0 = model.forward(params)

    return derive_desicions_from_taste_shock_peer_effects(n, gamma, thresholds_gamma_0, adj_m_peers, taste_shocks, report_adding)


"""

The adj_m is unidirectional in the sense
adj_m[i,j] = 1 means i is influenced by j

"""

def derive_desicions_from_taste_shock_peer_effects(n, gamma, thresholds_gamma_0, adj_m_peers, taste_shocks, report_adding=False):
    y_0 = np.zeros(n)
    y_1 = taste_shocks < thresholds_gamma_0
    y_org = np.array(y_1,dtype=np.int)
    while any(y_0 != y_1):
        y_0 = y_1
        n_peers_with_y_1 = np.matmul(adj_m_peers, y_1)
        y_1 = np.array(taste_shocks < thresholds_gamma_0 + gamma * n_peers_with_y_1,dtype=np.int)

    if (report_adding):
        print("added:  " + str(sum(y_1-y_org))+" decition")
    added_due_to_interaction = sum(y_1-y_org)

    return np.array(y_1,dtype=np.int), added_due_to_interaction