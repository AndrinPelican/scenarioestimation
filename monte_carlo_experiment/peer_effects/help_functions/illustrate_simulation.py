"""
This file illustrates the MC experiment

for Bryan:

    - Run 23
    - Run 24

"""

import pickle
import latextable
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats
from texttable import Texttable
from monte_carlo_experiment.monte_carlo_util.wald_test import wald_test

file = open("../saved_simulations/run_24.pickle", "rb")

pickle_dict = pickle.load(file)
params = pickle_dict["params"]
gamma= pickle_dict["gamma"]
print("Name of network: "+str(pickle_dict["name_network_creation"]) )
print("Comment: "+str(pickle_dict["comment"] if "comment" in pickle_dict else "no comment") )
print("N: "+str(pickle_dict["N"]) )
print("params: "+str(params) )
print("gamma: "+str(pickle_dict["gamma"]) )
print("n_buckets: "+str(pickle_dict["n_buckets"]) )
print("n_blocks: "+str(pickle_dict["n_blocks"]) )
info_dict_list = pickle_dict["info_dict_list"]
gamma_list = [my_dict["gamma_hat"] for my_dict in info_dict_list]


table_rows = [["N = "+str(pickle_dict["N"]),"" ]]
table_rows.append(["simulations = "+str(len(info_dict_list)),"" ])
table_rows.append(["runs = "+str(pickle_dict["n_runs"]),"" ])
# table_rows.append(["params = "+str(pickle_dict["params"],"" ])
table_rows.append(["buckets = "+str(pickle_dict["n_buckets"]), "gamma"])
table_rows.append(["mean", np.mean(gamma_list) ])
# table_rows.append(["Error mean", np.std(gamma_list)/np.sqrt(len(info_dict_list)) ])
# table_rows.append(["median", np.median(gamma_list) ])
table_rows.append(["Std. Dev.", np.std(gamma_list) ])

# table_rows.append(["Mean Std. Err.", mean_estimated_std ])
# see https://github.com/JAEarly/latextable for documentation on makeing table
table = Texttable()
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['a',
                      'a'
                      ]) # automatic
table.set_cols_align(["l", "r"])


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

The coverage tests:

"""

list_of_wald_qunatile = []
w_t_list = []
oben = 0
unten = 0
for my_dict in info_dict_list:
    w_r,wald_qunatile = wald_test(params_null=np.array([gamma] + list(params)), info_dict=my_dict)
    if w_r<2000:
        if (wald_qunatile>0.95):
            if (my_dict["gamma_hat"]>0.2):
                oben += 1
            else:
                unten += 1
        list_of_wald_qunatile.append(wald_qunatile)
        w_t_list.append(w_r)
        wald_test(params_null=np.array([gamma] + list(params)), info_dict=my_dict)
    else:
        print("invalite value probably gamma = 0")

print("oben: "+ str(oben))
print("unten: "+ str(unten))

# plt.hist(w_t_list, bins=30)
# plt.show()



# list_of_wald_qunatile = [ wald_test(params_null=np.array([gamma] + list(params)), info_dict=my_dict)[1] for my_dict in info_dict_list]
count_values_in_5p_range = 0
for value in list_of_wald_qunatile:
    if (value>0.95):
        count_values_in_5p_range += 1
print("Wald test: " + str(count_values_in_5p_range/len(list_of_wald_qunatile)))
table_rows.append(["Wald test (5%): ",str(count_values_in_5p_range/len(list_of_wald_qunatile)) ])

# stats.probplot(gamma_list, dist="norm", plot=pylab)
# pylab.show()
#
# plt.hist(list_of_wald_qunatile, bins=20)
# plt.show()



list_of_likelyhood_qunatile = [my_dict["quantile_lr"] for my_dict in info_dict_list]
count_values_in_5p_range = 0
for value in list_of_likelyhood_qunatile:
    if ( value>0.95 ):
        count_values_in_5p_range += 1
print("likelyhood test: " + str(count_values_in_5p_range/len(list_of_likelyhood_qunatile)))
table_rows.append(["likelyhood ratio test (5%): ",str(count_values_in_5p_range/len(list_of_likelyhood_qunatile)) ])

# plt.hist(list_of_likelyhood_qunatile, bins=20)
# plt.show()
#


count_values_in_5p_range = 0
for my_dict in info_dict_list:
    normalized = (my_dict["gamma_hat"]-gamma)/my_dict['std_errors'][0]
    if str(my_dict['std_errors'][0])=="nan":
        print("SSSSSSSSSSSSSSSSSSs")

    if (np.abs(normalized)>1.96):
        count_values_in_5p_range += 1

print("Standard error gamma coverage (5%): " + str(count_values_in_5p_range/len(info_dict_list)))
table_rows.append(["Standard error gamma coverage (5%): ",str(count_values_in_5p_range/len(list_of_likelyhood_qunatile)) ])




"""

The distribution of gamma:

"""

n_bins = 25
fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
axs.hist(gamma_list, bins=n_bins)
axs.axvline(np.mean(gamma_list), color='g')
axs.axvline(x=pickle_dict["gamma"], color='r')
axs.set_xlabel('gamma peer effects')
plt.show()

stats.probplot(gamma_list, dist="norm", plot=pylab)
pylab.show()

table.add_rows(table_rows)
print(table.draw() + "\n")
print('\nTexttable Table:')
print(latextable.draw_latex(table, caption="Monte Carlo Experiment", label="table:mc_experiment") + "\n")



