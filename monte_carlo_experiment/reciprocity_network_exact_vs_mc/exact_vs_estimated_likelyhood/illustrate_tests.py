import pickle

import numpy as np
from monte_carlo_experiment.monte_carlo_util.wald_test import wald_test
import matplotlib.pyplot as plt


file = open("infoDicts", "rb")
info_dict_list = pickle.load(file)

n = 10
# True parameters
params = np.array([-0.2])
gamma = 0.5
n_estimations = 2000


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
            if (my_dict["gamma_hat"]>0.5):
                oben += 1
            else:
                unten += 1
            print( my_dict["gamma_hat"]>0.2)
        list_of_wald_qunatile.append(wald_qunatile)
        w_t_list.append(w_r)
        wald_test(params_null=np.array([gamma] + list(params)), info_dict=my_dict)
    else:
        print("invalite value probably gamma = 0")

print("oben: "+ str(oben))
print("unten: "+ str(unten))

plt.hist(w_t_list, bins=30)
plt.show()



# list_of_wald_qunatile = [ wald_test(params_null=np.array([gamma] + list(params)), info_dict=my_dict)[1] for my_dict in info_dict_list]
count_values_in_5p_range = 0
for value in list_of_wald_qunatile:
    if (value>0.95):
        count_values_in_5p_range += 1
print("Wald test: " + str(count_values_in_5p_range/len(list_of_wald_qunatile)))

plt.hist(list_of_wald_qunatile, bins=180)
plt.show()




# list_of_wald_qunatile = [ wald_test(params_null=np.array([gamma] + list(params)), info_dict=my_dict)[1] for my_dict in info_dict_list]
count_values_in_5p_range = 0
likelyhood_quantiles = [my_dict['quantile_lr'] for my_dict in info_dict_list]
for value in likelyhood_quantiles:
    if (value>0.95):
        count_values_in_5p_range += 1
print("Likelyhood test: " + str(count_values_in_5p_range/len(likelyhood_quantiles)))

plt.hist(likelyhood_quantiles, bins=180)
plt.show()




count_values_in_5p_range = 0
for my_dict in info_dict_list:
    normalized = (my_dict["gamma_hat"]-gamma)/my_dict['std_errors'][0]
    if (np.abs(normalized)>1.96):
        count_values_in_5p_range += 1

print("Standard error out of range: " + str(count_values_in_5p_range/len(info_dict_list)))

