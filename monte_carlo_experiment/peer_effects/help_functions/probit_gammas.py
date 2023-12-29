import pickle
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.discrete.discrete_model as dm
from texttable import Texttable
from monte_carlo_experiment.monte_carlo_util.wald_test import wald_test

# see https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Probit.html


file = open("../saved_simulations/run_24.pickle", "rb")

pickle_dict = pickle.load(file)
params = pickle_dict["params"]
gamma= pickle_dict["gamma"]
print("Name of network: "+str(pickle_dict["name_network_creation"]) )
print("Comment: "+str(pickle_dict["comment"] if "comment" in pickle_dict else "no comment") )
print("N: "+str(pickle_dict["N"]) )
print("params: "+str(params) )
print("gamma: "+str(pickle_dict["gamma"]) )
print("n_blocks: "+str(pickle_dict["n_blocks"]) )
info_dict_list = pickle_dict["info_dict_list"]



"""

Calculate the gamma using classical probit

"""

def get_probit_gamma_list_from_pickle(pickle_dict):
    gamma_list = []
    X_original = pickle_dict["X"]
    adj_m = pickle_dict['adj_m']
    info_dict_list = pickle_dict["info_dict_list"]
    for my_dict in info_dict_list:
        y = my_dict["y"]
        X_1 = np.matmul(adj_m, y)[:,np.newaxis]
        X = np.concatenate([X_1, X_original], axis=1)

        probit_model = dm.Probit(y,X)
        result = probit_model.fit()
        gamma_list.append(result.params[0])

    return gamma_list




gamma_list = get_probit_gamma_list_from_pickle(pickle_dict)
table_rows = [["N = "+str(pickle_dict["N"]),"" ]]
table_rows.append(["simulations = "+str(len(info_dict_list)),"" ])
table_rows.append(["runs = "+str(pickle_dict["n_runs"]),"" ])
# table_rows.append(["params = "+str(pickle_dict["params"],"" ])
table_rows.append(["mean", np.mean(gamma_list) ])
table_rows.append(["error mean", np.std(gamma_list)/np.sqrt(len(info_dict_list)) ])

table_rows.append(["median", np.median(gamma_list) ])
table_rows.append(["Std. Dev.", np.std(gamma_list) ])
# table_rows.append(["Mean Std. Err.", mean_estimated_std ])
# see https://github.com/JAEarly/latextable for documentation on makeing table
table = Texttable()
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['a',
                      'a'
                      ]) # automatic
table.set_cols_align(["l", "r"])
table.add_rows(table_rows)
print(table.draw() + "\n")

"""
Density distribution!

"""



# i = 0
# indexes_to_remove = []
# for ind, my_dict in enumerate(info_dict_list):
#     sdf =my_dict['std_errors'][0]
#     if str(my_dict['std_errors'][0])=="nan":
#         i += 1
#         print(i)
#         indexes_to_remove.append(ind)


# for index in sorted(indexes_to_remove, reverse=True):
#     del info_dict_list[index]



"""

The distribution of gamma:

"""
#
# n_bins = 25
# fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
# axs.hist(gamma_list, bins=n_bins)
# axs.axvline(np.mean(gamma_list), color='g')
# axs.axvline(x=pickle_dict["gamma"], color='r')
# axs.set_xlabel('gamma peer effects')
# plt.show()