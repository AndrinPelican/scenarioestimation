"""
This is a monte carlo experiment

Goals is simulate graphs with density parameter a = -0.2 and then S is reciprocity with gamma = 0.5

From the graphs estimate a_hat and gamma_hat

"""

import matplotlib.pyplot as plt
import numpy as np

from estimation.estimators.network.estimate_model_network import estimate_model_network
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel
from util.graph_functions.random_digraph import create_digraph_reciprocity
from util.graph_functions.s_functions import s_reciprocity

n = 20
# True parameters
params = np.array([-0.2])
gamma = 0.5

erdoes_reny_model = ErdoesRenyModel(n)
a_hats = []
gamma_hats = []
denitys = []
for i in range(100):
    print("==================================================================" + str(i))
    adj_m = create_digraph_reciprocity(n, params=params, model=erdoes_reny_model, gamma=gamma)
    denitys.append(sum(sum(adj_m)) / (n * (n - 1)))

    gamma_hat, a_hat, info_dict = estimate_model_network(adj_m=adj_m,
                                                         model=erdoes_reny_model,
                                                         s_function=s_reciprocity,
                                                         n_buckets=10,
                                                         n_runs=4)

    # gamma_hat, a_hat, info_dict = estimate_model_reciprocity_exact(adj_m=adj_m,model=erdoes_reny_model)
    print("a_hat  " + str(a_hat))
    print("gamma hat " + str(gamma_hat))
    gamma_hats.append(gamma_hat)
    a_hats.append(a_hat[0])

print("done")
print(np.mean(gamma_hats))
print(np.mean(a_hats))
print(np.mean(denitys))

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
n_bins = 20
# We can set the number of bins with the `bins` kwarg
axs[0].hist(gamma_hats, bins=n_bins)
axs[0].axvline(x=gamma, color='r')
axs[0].set_xlabel('gamma')

axs[1].hist(a_hats, bins=n_bins)
axs[1].axvline(x=params[0], color='r')
axs[1].set_xlabel('a')

axs[2].hist(denitys, bins=n_bins)
axs[2].set_xlabel('denitys')

plt.show()
