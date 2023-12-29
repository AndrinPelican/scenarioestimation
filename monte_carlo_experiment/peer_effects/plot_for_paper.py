
import pickle
import numpy as np

import matplotlib.pyplot as plt
from monte_carlo_experiment.peer_effects.help_functions.probit_gammas import get_probit_gamma_list_from_pickle

"""
     Loading the many medium sized games
"""

file = open("./saved_simulations/run_23.pickle", "rb")
pickle_dict = pickle.load(file)
print("n_blocks: "+str(pickle_dict["n_blocks"]) )
gamma_many= pickle_dict["gamma"]
info_dict_list = pickle_dict["info_dict_list"]
gamma_list_many = [my_dict["gamma_hat"] for my_dict in info_dict_list]
gamma_list_probit_many = get_probit_gamma_list_from_pickle(pickle_dict)








"""
     Loading the single big game
"""

file = open("saved_simulations/run_24.pickle", "rb")
pickle_dict = pickle.load(file)
print("n_blocks: "+str(pickle_dict["n_blocks"]) )
gamma_single= pickle_dict["gamma"]
info_dict_list = pickle_dict["info_dict_list"]
gamma_list_single = [my_dict["gamma_hat"] for my_dict in info_dict_list]
gamma_list_probit_single = get_probit_gamma_list_from_pickle(pickle_dict)




"""

The distribution of gamma:

"""


fig, axs = plt.subplots(1, 2, sharey="all", tight_layout=True)
bins = np.linspace(-0.15, 0.5, 50)
axs[0].hist(gamma_list_many, bins, alpha=0.6, density=True, label='scenario estimation', color="blue")
axs[0].hist(gamma_list_probit_many, bins, alpha=0.3, density=True, label='probit', color="g")
axs[0].axvline(np.mean(gamma_list_many),  color="blue")
axs[0].axvline(np.mean(gamma_list_probit_many),  color="g")
axs[0].axvline(x=gamma_many, color='r')
axs[0].legend(loc='upper right')
axs[0].set_title("Many Medium Sized Games")


axs[1].hist(gamma_list_single, bins, alpha=0.6, density=True,label='scenario estimation', color="blue")
axs[1].hist(gamma_list_probit_single, bins, alpha=0.3, density=True,label='probit', color="g")
axs[1].axvline(np.mean(gamma_list_single),  color="blue")
axs[1].axvline(np.mean(gamma_list_probit_single),  color="g")
axs[1].axvline(x=gamma_single, color='r')
axs[1].legend(loc='upper right')
axs[1].set_title("Single Big Game")
plt.show()
# plt.style.use('seaborn-deep')
#
# axs.hist([gamma_list,gamma_list_probit], density=True, bins=n_bins)
# # axs.hist(gamma_list_probit, bins=n_bins, density=True, color="yellow")
# axs.axvline(np.mean(gamma_list), color='g')
# axs.axvline(x=pickle_dict["gamma"], color='r')
# axs.set_xlabel('gamma peer effects')
# plt.show()

