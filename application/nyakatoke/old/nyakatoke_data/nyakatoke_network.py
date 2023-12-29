import numpy as np
import pandas as pd

# The Data
from util.params_and_X_for_degree_heterogeneity import X_matrix_fixed_effect_for_network_n_agents

Nyakatoke_ind = pd.read_stata("./nyakatoke_data/Nyakatoke individual.dta")
Nyakatoke_dyad = pd.read_stata("./nyakatoke_data/Nyakatoke dyadic.dta")
Nyakatoke_hh = pd.read_stata("./nyakatoke_data/Nyakatoke household.dta")
Nyakatoke_dir = pd.read_stata("./nyakatoke_data/Nyakatoke directed.dta")

# 91, and 63 have no distance information
indexes_to_drop = {91.0, 63.0, 47.}

var_dict = {}
nyakatoke_id_to_id = {}
id_to_nyakatoke_id = {}
# nyakatoke id 116 does not exist
i = 0
for _, nyakatoke_id in enumerate(Nyakatoke_dyad['hh1'].unique()):
    if nyakatoke_id in indexes_to_drop:
        continue
    nyakatoke_id_to_id[nyakatoke_id] = i
    id_to_nyakatoke_id[i] = nyakatoke_id
    i += 1
# add 122, it is not in Nyakatoke_dyad['hh1'].unique(), because it is the last link
nyakatoke_id_to_id[122.0] = i
id_to_nyakatoke_id[i] = 122.0

# adjacency matrix:
n_agents = nyakatoke_id_to_id.__len__()
n_arrows = n_agents * (n_agents - 1)
di_adj_m = np.zeros(shape=(n_agents, n_agents))

for key, row in Nyakatoke_dir.iterrows():
    if row["hh1"] in indexes_to_drop or row["hh2"] in indexes_to_drop:
        continue
    from_int = nyakatoke_id_to_id[row['hh1']]
    to_int = nyakatoke_id_to_id[row['hh2']]
    di_adj_m[from_int, to_int] = 1

'''
Create adj _m and variable_dict
'''


def distance():
    X = - np.ones((n_arrows, 1))
    # filling it up
    for key, row in Nyakatoke_dyad.iterrows():
        if row["hh1"] in indexes_to_drop or row["hh2"] in indexes_to_drop:
            continue
        id_from = nyakatoke_id_to_id[row["hh1"]]
        id_to = nyakatoke_id_to_id[row["hh2"]]
        distance = np.log(row["distance"])

        X[get_flatten_ind(id_from, id_to), 0] = distance
        X[get_flatten_ind(id_to, id_from), 0] = distance
    return X


def same_clan():
    X = - np.ones((n_arrows, 1))
    # filling it up
    for key, row in Nyakatoke_dyad.iterrows():
        if row["hh1"] in indexes_to_drop or row["hh2"] in indexes_to_drop:
            continue
        id_from = nyakatoke_id_to_id[row["hh1"]]
        id_to = nyakatoke_id_to_id[row["hh2"]]
        is_same_clan = 1 if row["clan1"] == row["clan2"] else 0
        X[get_flatten_ind(id_from, id_to), 0] = is_same_clan
        X[get_flatten_ind(id_to, id_from), 0] = is_same_clan

    return X


def same_religion():
    X = - np.ones((n_arrows, 1))
    # filling it up
    for key, row in Nyakatoke_dyad.iterrows():
        if row["hh1"] in indexes_to_drop or row["hh2"] in indexes_to_drop:
            continue
        id_from = nyakatoke_id_to_id[row["hh1"]]
        id_to = nyakatoke_id_to_id[row["hh2"]]

        is_same_religin = 1 if row["religion1"] == row["religion2"] else 0
        X[get_flatten_ind(id_from, id_to), 0] = is_same_religin
        X[get_flatten_ind(id_to, id_from), 0] = is_same_religin
    return X


def create_covariates_for_nyakatoke(use_fixed_effects_instead_of_constant=False, use_distance=False,
                                    use_same_religion=False, use_same_clan=False):
    covariate_string = ""
    if (use_fixed_effects_instead_of_constant):
        X = X_matrix_fixed_effect_for_network_n_agents(n_agents)
        covariate_string = " fixed_effect_constants "

        # there are a few notes with no in links/ out links, we remove the correspoinding columns
        X = np.delete(X, [28, 121, 150, 158, 196, 207, 218, 221, 227, 228, 229, 230], axis=1)




    else:
        X = np.ones((n_arrows, 1))
        covariate_string = " constant "

    if use_distance:
        X_distance = distance()
        X = np.concatenate([X_distance, X], axis=1)
        covariate_string = " distance " + covariate_string

    if use_same_clan:
        X_clan = same_clan()
        X = np.concatenate([X_clan, X], axis=1)
        covariate_string = " same_clan " + covariate_string

    if use_same_religion:
        X_religion = same_religion()
        X = np.concatenate([X_religion, X], axis=1)
        covariate_string = " same_religion " + covariate_string

    check_colliniarity(X)

    print(" you use as covariats: " + covariate_string)
    return X


def check_colliniarity(X):
    if not (np.linalg.matrix_rank(X) == X.shape[1]):
        print("attention, perfekt coliniarity!")
        assert False


def get_flatten_ind(id_from, id_to):
    # get the right index to write it in, considering the 0 os in the adj matrix
    shift_do_consider_diag = + -1 if id_to > id_from else 0
    k = id_from * (n_agents - 1) + id_to + shift_do_consider_diag
    return k
