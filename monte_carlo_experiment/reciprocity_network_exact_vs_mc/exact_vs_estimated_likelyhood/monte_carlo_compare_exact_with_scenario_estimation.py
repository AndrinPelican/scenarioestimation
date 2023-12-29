"""

Scenario Estimation is a random estimation methods, because the scenarios are sampled.

This file has as purpose to see how this random is in comparison to the exact estimator.

We use Reciprocity in nyakatoke, because we can evaluate the likelihood closed form.

"""

import matplotlib.pyplot as plt
import numpy as np

from monte_carlo_experiment.reciprocity_network_exact_vs_mc.estimate_model_reciprocity_considering_subgames import \
    estimate_reciprocyty_subgames
from monte_carlo_experiment.reciprocity_network_exact_vs_mc.estimate_model_reciprocity_exact import \
    estimate_model_reciprocity_exact
from estimation.model.erdoes_reny_base_model import ErdoesRenyModel

n = 8
# True parameters
params = np.array([-0.2])
gamma = 0.5
n_estimations = 20

erdoes_reny_model = ErdoesRenyModel(n)

gamma_hats_1_bucket = []
gamma_hats_10_bucket = []
gamma_subgames_indpendent = []
gamma_subgames_indpendent_biased_corrected = []
gamma_subgames_indpendent_unbiased = []

"""
The adj matrix has been created with:
    adj_m = create_digraph_reciprocity(n, params= params, model=erdoes_reny_model, gamma=gamma)

in order to ensure reproducibility, we wrote the simulated matrix directly into the code.
"""
adj_m = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])

# adj_m = create_digraph_reciprocity(n, params= params, model=erdoes_reny_model, gamma=gamma)

gamma_hat_exact, a_hat, info_dict = estimate_model_reciprocity_exact(adj_m=adj_m, model=erdoes_reny_model)

for i in range(n_estimations):
    print("================================================================== indpendent " + str(i))
    gamma_hat, a_hat, info_dict = estimate_reciprocyty_subgames(adj_m=adj_m, n_buckets=10, n_runs=2, type="indpendent")
    gamma_subgames_indpendent.append(gamma_hat)

# for i in range(n_estimations):
#     print("================================================================== indpendent_biased_corrected "+str(i))
#     gamma_hat, a_hat, info_dict = estimate_reciprocyty_subgames(adj_m=adj_m, n_buckets=10, n_runs=2, type="indpendent_biased_corrected")
#     gamma_subgames_indpendent_biased_corrected.append(gamma_hat)


for i in range(n_estimations):
    print("================================================================== indpendent_biased_corrected " + str(i))
    gamma_hat, a_hat, info_dict = estimate_reciprocyty_subgames(adj_m=adj_m, n_buckets=10, n_runs=2,
                                                                type="indpendent_unbiased")
    gamma_subgames_indpendent_unbiased.append(gamma_hat)

    # for i in range(n_estimations):
    #     print("=================================================================="+str(i))
    #     gamma_hat, a_hat, info_dict = estimate_model_network(adj_m = adj_m,
    #                                               model = erdoes_reny_model,
    #                                               from_shock_to_bucket_monotone_increasing = from_shock_to_bucket_reciprocity_considering_noLink,
    #                                               s_function=s_reciprocity,
    #                                               n_buckets=1,
    #                                               n_runs=2)
    #     gamma_hats_1_bucket.append(gamma_hat)
    #
    # for i in range(n_estimations):
    #     print("=================================================================="+str(i))
    #     gamma_hat, a_hat, info_dict = estimate_model_network(adj_m = adj_m,
    #                                               model = erdoes_reny_model,
    #                                               from_shock_to_bucket_monotone_increasing = from_shock_to_bucket_reciprocity_considering_noLink,
    #                                               s_function = s_reciprocity,
    #                                               n_buckets = 10,
    #                                               n_runs= 2)
    gamma_hats_10_bucket.append(gamma_hat)

print("done")
print("One Bucket")
print("std of estimations:  " + str(np.std(gamma_hats_1_bucket)))
print("bias of estimations: " + str(np.mean(gamma_hats_1_bucket) - gamma_hat_exact))

print("10 Bucket")
print("std of estimations:  " + str(np.std(gamma_hats_10_bucket)))
print("bias of estimations: " + str(np.mean(gamma_hats_10_bucket) - gamma_hat_exact))

print("gamma_subgames_indpendent_biased_corrected ")
print("std of estimations:  " + str(np.std(gamma_subgames_indpendent_biased_corrected)))
print("bias of estimations: " + str(np.mean(gamma_subgames_indpendent_biased_corrected) - gamma_hat_exact))

print("gamma_subgames gamma_subgames_indpendent ")
print("std of estimations:  " + str(np.std(gamma_subgames_indpendent)))
print("bias of estimations: " + str(np.mean(gamma_subgames_indpendent) - gamma_hat_exact))

print("UNBIASED")
print("std of estimations:  " + str(np.std(gamma_subgames_indpendent_unbiased)))
print("bias of estimations: " + str(np.mean(gamma_subgames_indpendent_unbiased) - gamma_hat_exact))

fig, axs = plt.subplots(1, 5, sharey=True, tight_layout=True)
n_bins = 20
# We can set the number of bins with the `bins` kwarg
axs[0].hist(gamma_hats_1_bucket, bins=n_bins)
axs[0].axvline(x=gamma_hat_exact, color='g')
axs[0].set_xlabel('1 BUCKETS')

axs[1].hist(gamma_hats_10_bucket, bins=n_bins)
axs[1].axvline(x=gamma_hat_exact, color='g')
axs[1].set_xlabel('10 BUCKETS')

axs[2].hist(gamma_subgames_indpendent, bins=n_bins)
axs[2].axvline(x=gamma_hat_exact, color='g')
axs[2].set_xlabel('INDEPENDENT')

axs[3].hist(gamma_subgames_indpendent_biased_corrected, bins=n_bins)
axs[3].axvline(x=gamma_hat_exact, color='g')
axs[3].set_xlabel('BIAS CORRECTED')

axs[4].hist(gamma_subgames_indpendent_unbiased, bins=n_bins)
axs[4].axvline(x=gamma_hat_exact, color='g')
axs[4].set_xlabel('UNBIASED')

plt.show()
