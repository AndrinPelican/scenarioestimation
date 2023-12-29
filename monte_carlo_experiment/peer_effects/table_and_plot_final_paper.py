import pickle
import latextable
import numpy as np
from texttable import Texttable


file_names= [
    "all__20_player_per_game__100_indeptendnt_games__1_szenarios.pickle",
    "all__20_player_per_game__100_indeptendnt_games__10_szenarios.pickle",
    "all__20_player_per_game__100_indeptendnt_games__100_szenarios.pickle",
    "all__500_player_per_game__1_indeptendnt_games__1_szenarios.pickle",
    "all__500_player_per_game__1_indeptendnt_games__10_szenarios.pickle",
    "all__500_player_per_game__1_indeptendnt_games__100_szenarios.pickle",
]

row_names = [
    r"Number of games",
    r"Number of players in one game",
    r"Number of scenario samples",
    r"Number Monte Carlo replications",
    r"Mean of $\hat{\delta}$",
    r"Std. Dev. of $\hat{\delta}$",
    r"Likelihood Ratio test size ($H_0 : \delta = 0, \alpha = 5\%$)",
    r"Confidence interval coverage ($\alpha = 5\%$)"
]

def make_list_table_values(file_name:str):
    file = open("./saved_simulations/"+file_name, "rb")
    pickle_dict = pickle.load(file)
    gamma= pickle_dict["gamma"]
    info_dict_list = pickle_dict["info_dict_list"]
    gamma_list = [my_dict["gamma_hat"] for my_dict in info_dict_list]
    print("players total: "+ str(pickle_dict["n_agents_total"]))
    print("in indipendent gammes: " +str(int(pickle_dict['n_agents_total']/pickle_dict["block_size"])))

    # Coverage:
    list_of_likelyhood_qunatile = [my_dict["quantile_lr"] for my_dict in info_dict_list]
    count_values_in_5p_range = 0
    for value in list_of_likelyhood_qunatile:
        if ( value>0.95 ):
            count_values_in_5p_range += 1
    size_likelyhood =  count_values_in_5p_range/len(list_of_likelyhood_qunatile)


    count_values_in_5p_range = 0
    for my_dict in info_dict_list:
        normalized = (my_dict["gamma_hat"]-gamma)/my_dict['std_errors'][0]
        if str(my_dict['std_errors'][0])=="nan":
            print("SSSSSSSSSSSSSSSSSSs")

        if (np.abs(normalized)>1.96):
            count_values_in_5p_range += 1
    coverage_std = 1- count_values_in_5p_range/len(info_dict_list)
    return  [ int(pickle_dict['n_agents_total']/pickle_dict["block_size"]), pickle_dict["block_size"],pickle_dict["n_szenarios"],len(gamma_list),np.mean(gamma_list), np.std(gamma_list),size_likelyhood, coverage_std]


result_list_of_list = [make_list_table_values(name) for name in file_names]
"""

The distribution of gamma:

"""
table_rows = []
for i in range(len(row_names)):
    table_rows.append([row_names[i]]+[l[i] for l in result_list_of_list])


# table_rows.append(["Mean Std. Err.", mean_estimated_std ])
# see https://github.com/JAEarly/latextable for documentation on makeing table
table = Texttable()
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['a','a','a', 'a','a','a', 'a'  ]) # automatic
table.set_cols_align(["l", "r", "r", "r", "r", "r", "r"])

table.add_rows(table_rows)
print('\nTexttable Table:')
print(latextable.draw_latex(table, caption="Monte Carlo Experiment", label="table:mc_experiment") + "\n")

