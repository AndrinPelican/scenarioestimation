import numpy as np
import statsmodels.api as sm
import pandas as pd
from application.nyakatoke.Nyakatoke_data import load_nyakatoke_network
from tabulate import tabulate
from estimation.estimators.network.estimate_model_network import estimate_model_network
from estimation.model.model_linear import LinearModel
from util.graph_functions.s_functions import s_support, s_transitivity

s_support
s_transitivity
runs = 4
s_funcion = s_support
n_buckets_list = [100,100,100]
# n_buckets_list = [1,1,1]
use_sender_reciver_effect = True

adjacency_matrix, variables, X, Y, Y_In_X = load_nyakatoke_network(use_sender_reciver_effect=use_sender_reciver_effect, s_function = s_funcion)

variables = [s_funcion.__name__]+variables

"""First probit regression: """

X_for_reg = np.concatenate([Y_In_X, X], axis=1)
# Fit probit regression using statsmodels
probit_model = sm.Probit(Y, X_for_reg)
result = probit_model.fit()

# Get the coefficient estimates and standard errors
coefs_naive = result.params
se_naive = result.bse


# Obtain the Hessian matrix
hessian = result.mle_retvals["Hessian"]

# Calculate the condition number of the Hessian matrix
condition_number = np.linalg.cond(hessian)
print("Condition number Hessian:", condition_number)

# Create a table with variable names, regression coefficients, and standard errors
table_data = {'Variable': variables, 'Coefficient': coefs_naive, 'Standard Error': se_naive}
table = pd.DataFrame(table_data)
print(table)

# Obtain the Hessian matrix
hessian = result.mle_retvals["Hessian"]

# Calculate the condition number of the Hessian matrix
condition_number = np.linalg.cond(hessian)
print("Condition number Hessian:", condition_number)


coef_szenario_list = []
sd_szenario_list = []
for i in range(3):
    linearModel = LinearModel(X)
    linearModel.initial_gamma = coefs_naive[0]
    linearModel.initial_params = coefs_naive[1:]

    gamma_had, a_hat, info_dict = estimate_model_network(adj_m=adjacency_matrix, model=linearModel,  s_function=s_funcion, n_buckets=n_buckets_list[i], n_runs=runs)
    coef_szenario_list.append(info_dict["all_params_hat"])
    sd_szenario_list.append(info_dict["std_errors"])


# Create a list of lists with the data
table_data = []
for var_name, c1, se1, c2, se2, c3,se3, c4, se4 in zip(variables, coefs_naive, se_naive,
                                                       coef_szenario_list[0], sd_szenario_list[0],
                                                       coef_szenario_list[1], sd_szenario_list[1],
                                                       coef_szenario_list[2], sd_szenario_list[2]):
    v1 = f"{c1:.3f}  ({se1:.3f})"
    v2 = f"{c2:.3f}  ({se2:.3f})"
    v3 = f"{c3:.3f}  ({se3:.3f})"
    v4 = f"{c4:.3f}  ({se4:.3f})"
    if "fe_" in var_name:
        continue
    table_data.append([var_name, v1, v2, v3, v4])

if use_sender_reciver_effect:
    table_data.append(["fixed effects", "True","True","True","True"])

# Define table headers
headers = ["Variable", "probit", f" {n_buckets_list[0]} szenarios ", f" {n_buckets_list[1]} szenarios ", f" {n_buckets_list[2]} szenarios "]
# Print the table
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Print the table using tabulate
table = tabulate(table_data, headers=headers, tablefmt="latex")
# Print the LaTeX table
print("\\begin{table}\n \\begin{center}")
print(table)
print("\end{center} \\caption{Estiation of support in the Nyakatoke. The fist culums is the probit esitmate, which is known to be biased. "
      "The following rows are the Szenario estimation estimate, for a variing number of szenarios used for the MLE simulation. (runs: "+str(runs)+")} \\label{table:another_table}" \
"\\end{table}")
