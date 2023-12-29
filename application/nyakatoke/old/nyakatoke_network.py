"""

This file runs the ugd algorithm on the Nyakatoke nyakatoke in order to

- compare it result with other type of algorithms like importance create_frendschip_network
- see whether considering other constraints  like wealth has an impact on the test results

"""
import latextable
from texttable import Texttable

from application.nyakatoke.old.nyakatoke_data.nyakatoke_network import create_covariates_for_nyakatoke, di_adj_m
from estimation.estimators.network.estimate_model_network import estimate_model_network
from estimation.model.model_linear import LinearModel
from util.graph_functions.s_functions import s_support, s_transitivity

s_support
s_transitivity

# loading the Network
M = di_adj_m
n = M.shape[0]

n_buckets = 1
n_runs = 3
table_rows = [
    [s_transitivity.__name__, "distance", "same religion", "same clan", "used fixed effects", "used constant"]]
table_rows.append(["", "", "", "", "", ""])
s_funcion = s_transitivity
"""

  Only density/constant

"""

# Determining model (what to control for)
X = create_covariates_for_nyakatoke(use_distance=False, use_same_clan=False, use_same_religion=False,
                                    use_fixed_effects_instead_of_constant=False)
linearModel = LinearModel(X)
linearModel.initial_gamma = 0.5
linearModel.initial_params = [-1.904]

gamma_hat_05Start, a_hat, info_dict = estimate_model_network(adj_m=M, model=linearModel,
                                                             s_function=s_funcion, n_buckets=n_buckets, n_runs=n_runs)

table_rows.append([gamma_hat_05Start, "", "", "", "False", "True"])
table_rows.append(["({:.3f})".format(info_dict["std_errors"][0]), "", "", "", "", ""])
table_rows.append(["", "", "", "", "", ""])

"""

  Only Fixed effects

"""

# Determining model (what to control for)
X = create_covariates_for_nyakatoke(use_distance=False, use_same_clan=False, use_same_religion=False,
                                    use_fixed_effects_instead_of_constant=True)
linearModel = LinearModel(X)
linearModel.initial_gamma = 0.39
linearModel.initial_params = [-1.625, -1.780, -1.409, -1.827, -1.733, -1.933, -1.551, -1.704, -1.242, -2.209, -1.857,
                              -1.841, -1.457, -1.677, -1.972, -1.629, -1.872, -1.466, -1.610, -1.820, -1.644, -1.447,
                              -1.618, -1.730, -1.414, -1.823, -1.591, -1.241, -1.698, -1.812, -1.487, -1.580, -1.638,
                              -1.941, -1.988, -1.466, -1.735, -1.585, -1.575, -1.532, -1.621, -1.516, -1.617, -1.891,
                              -1.496, -1.634, -1.794, -1.834, -1.697, -1.641, -1.556, -1.443, -1.424, -1.586, -1.230,
                              -1.844, -1.493, -1.540, -1.464, -1.477, -1.789, -1.802, -1.553, -1.984, -1.611, -1.368,
                              -1.615, -1.703, -1.965, -1.604, -1.587, -1.572, -1.702, -1.617, -1.699, -1.828, -1.602,
                              -1.834, -1.783, -1.728, -1.960, -1.855, -1.972, -1.805, -1.835, -1.538, -1.663, -1.930,
                              -1.461, -1.658, -1.984, -1.486, -1.708, -1.709, -1.433, -1.709, -1.829, -1.520, -1.669,
                              -1.679, -1.455, -2.238, -1.384, -1.647, -1.977, -1.590, -1.923, -1.637, -1.566, -1.743,
                              -2.004, -1.689, -1.804, -1.743, -0.244, -0.474, -0.300, -0.402, -0.368, -0.598, -0.215,
                              -0.597, 0.277, -0.143, -0.506, -0.384, -0.901, -0.561, -0.613, 0.219, -0.115, -0.060,
                              0.029, 0.108, -0.220, -0.339, -0.622, -0.579, -0.190, -0.366, -0.187, 0.087, -0.310,
                              0.090, -0.434, -0.207, 0.014, -0.074, -0.155, -0.175, -0.108, -0.330, -0.255, -0.193,
                              -0.503, -0.031, -0.324, -0.176, -0.234, -0.326, -0.310, -0.505, -0.413, 0.099, -0.476,
                              -0.219, -0.178, 0.311, -0.309, -0.369, -0.068, -0.573, -0.280, -0.911, -0.405, -0.867,
                              -0.886, -0.288, -0.152, -0.017, -0.006, -0.323, 0.038, -0.271, -0.288, -0.608, -0.293,
                              -0.607, -0.903, -0.343, -0.584, -0.276, -0.603, -0.022, -0.285, -0.337, -0.413, -0.301,
                              -0.224, -0.499, -0.519, -0.261, -0.204, -0.634, -0.258, -0.260, -0.144, -0.330, -0.321,
                              -0.419, -0.139, -0.465, 0.127, -0.141, -0.211, 0.084, -0.599, -0.015, -0.693]

gamma_hat_05Start, a_hat, info_dict = estimate_model_network(adj_m=M, model=linearModel,
                                                             s_function=s_funcion, n_buckets=n_buckets, n_runs=n_runs)

table_rows.append([gamma_hat_05Start, "", "", "", "True", "False"])
table_rows.append(["({:.3f})".format(info_dict["std_errors"][0]), "", "", "", "", ""])
table_rows.append(["", "", "", "", "", ""])

# Determining model (what to control for)
X = create_covariates_for_nyakatoke(use_distance=True, use_same_clan=False, use_same_religion=False,
                                    use_fixed_effects_instead_of_constant=True)
linearModel = LinearModel(X)
linearModel.initial_gamma = 0.316
linearModel.initial_params = [-0.630, 0.294, 0.121, 0.674, -0.006, 0.254, 0.242, 0.376, 0.168, 0.907, -0.161, -0.051,
                              -0.023, 0.428, 0.120, -0.243, 0.435, -0.007, 0.467, 0.231, 0.015, 0.140, 0.394, 0.242,
                              -0.062, 0.398, -0.068, 0.264, 0.652, 0.295, 0.056, 0.393, 0.195, 0.106, -0.233, -0.301,
                              0.310, 0.095, 0.145, 0.140, 0.208, 0.238, 0.506, 0.427, -0.153, 0.168, 0.225, 0.064,
                              -0.178, 0.112, 0.081, 0.202, 0.388, 0.386, 0.145, 0.706, -0.198, 0.367, 0.266, 0.391,
                              0.469, 0.085, -0.070, 0.237, -0.211, 0.308, 0.626, -0.015, 0.107, -0.024, 0.380, 0.063,
                              0.432, 0.366, 0.102, 0.120, -0.237, 0.175, -0.158, 0.093, 0.379, 0.208, -0.021, -0.378,
                              -0.107, 0.558, 0.317, 0.427, 0.234, 0.916, 0.481, 0.181, 0.705, 0.681, 0.187, 0.381,
                              0.068, 0.046, 0.765, 0.606, 0.708, 0.838, -0.304, 0.525, 0.146, -0.140, 0.253, 0.028,
                              0.260, 0.528, 0.098, -0.256, 0.129, 0.002, 0.506, 1.572, 1.329, 1.640, 1.375, 1.409,
                              1.237, 1.734, 1.257, 2.251, 1.944, 1.365, 1.450, 0.955, 1.162, 1.148, 2.373, 1.712, 1.721,
                              1.835, 1.984, 1.511, 1.379, 1.096, 0.848, 1.557, 1.351, 1.615, 1.852, 1.325, 2.038, 1.394,
                              1.467, 1.739, 1.587, 1.483, 1.516, 1.638, 1.298, 1.292, 1.467, 1.315, 1.900, 1.382, 1.531,
                              1.519, 1.493, 1.430, 1.183, 1.257, 1.913, 1.188, 1.498, 1.571, 2.059, 1.332, 1.396, 1.636,
                              1.214, 1.579, 0.823, 1.308, 0.788, 0.795, 1.395, 1.721, 1.733, 1.775, 1.465, 2.033, 1.417,
                              1.651, 1.114, 1.201, 1.109, 0.616, 1.410, 0.877, 1.492, 1.515, 1.725, 1.283, 1.355, 1.623,
                              1.402, 1.843, 1.565, 1.602, 1.593, 1.845, 1.254, 1.547, 1.505, 1.528, 1.605, 1.800, 1.618,
                              2.017, 1.614, 2.112, 1.664, 1.544, 1.990, 1.258, 2.023, 1.073]

gamma_hat_05Start, a_hat, info_dict = estimate_model_network(adj_m=M, model=linearModel,
                                                             s_function=s_funcion, n_buckets=n_buckets, n_runs=n_runs)

table_rows.append([gamma_hat_05Start, a_hat[0], "", "", "True", "False"])
table_rows.append(
    ["({:.3f})".format(info_dict["std_errors"][0]), "({:.3f})".format(info_dict["std_errors"][1]), "", "", "", ""])
table_rows.append(["", "", "", "", "", ""])

"""

Distance

"""

# Determining model (what to control for)
X = create_covariates_for_nyakatoke(use_distance=True, use_same_clan=True, use_same_religion=True,
                                    use_fixed_effects_instead_of_constant=True)
linearModel = LinearModel(X)
linearModel.initial_gamma = 0.337
linearModel.initial_params = [0.240, 0.359, -0.611, 0.223, 0.056, 0.582, 0.060, 0.091, 0.170, 0.325, 0.184, 0.857,
                              -0.177, -0.209, -0.115, 0.333, 0.015, -0.234, 0.504, -0.103, 0.308, 0.271, -0.084, 0.189,
                              0.445, 0.239, -0.159, 0.411, -0.168, 0.192, 0.604, 0.241, -0.017, 0.310, 0.142, 0.158,
                              -0.211, -0.303, 0.239, 0.090, 0.021, 0.168, 0.142, 0.228, 0.482, 0.367, -0.113, 0.110,
                              0.209, 0.016, -0.102, 0.049, 0.105, 0.131, 0.311, 0.393, -0.031, 0.589, -0.186, 0.220,
                              0.268, 0.361, 0.293, 0.071, -0.208, 0.139, -0.210, 0.188, 0.664, 0.030, -0.005, -0.008,
                              0.370, -0.118, 0.295, 0.307, 0.057, 0.123, -0.130, 0.188, -0.182, 0.062, 0.356, 0.158,
                              -0.054, -0.420, -0.196, 0.454, 0.234, 0.458, 0.220, 0.797, 0.461, 0.211, 0.677, 0.509,
                              0.124, 0.277, -0.030, 0.026, 0.719, 0.517, 0.708, 0.690, -0.300, 0.326, 0.002, -0.229,
                              0.197, 0.057, 0.148, 0.462, -0.010, -0.342, 0.181, -0.030, 0.448, 1.408, 1.122, 1.355,
                              1.145, 1.079, 0.711, 1.426, 1.133, 2.034, 1.773, 1.090, 1.091, 0.715, 0.888, 1.004, 2.154,
                              1.454, 1.438, 1.552, 1.723, 1.379, 1.183, 0.845, 0.548, 1.429, 1.079, 1.372, 1.560, 1.111,
                              1.777, 1.275, 1.288, 1.549, 1.371, 1.266, 1.227, 1.437, 1.031, 1.185, 1.282, 1.060, 1.650,
                              1.198, 1.258, 1.329, 1.243, 1.309, 0.916, 1.101, 1.710, 0.962, 1.287, 1.259, 1.869, 1.191,
                              1.167, 1.481, 0.943, 1.255, 0.586, 1.037, 0.543, 0.643, 1.230, 1.578, 1.582, 1.464, 1.325,
                              1.830, 1.206, 1.376, 1.009, 1.034, 0.933, 0.518, 1.221, 0.676, 1.329, 1.290, 1.518, 1.159,
                              1.087, 1.336, 1.051, 1.672, 1.377, 1.346, 1.401, 1.588, 1.037, 1.395, 1.284, 1.285, 1.303,
                              1.554, 1.376, 1.876, 1.372, 1.782, 1.313, 1.274, 1.748, 1.095, 1.775, 0.844]

gamma_hat_05Start, a_hat, info_dict = estimate_model_network(adj_m=M, model=linearModel,
                                                             s_function=s_funcion, n_buckets=n_buckets, n_runs=n_runs)

table_rows.append([gamma_hat_05Start, a_hat[2], a_hat[0], a_hat[1], "True", "False"])
table_rows.append(["({:.3f})".format(info_dict["std_errors"][0]), "({:.3f})".format(info_dict["std_errors"][3]),
                   "({:.3f})".format(info_dict["std_errors"][1]), "({:.3f})".format(info_dict["std_errors"][2]), "",
                   ""])
table_rows.append(["", "", "", "", "", ""])

"""

  Drawing

"""

print("\n\n")
print("gamma_hat_05Start")

# see https://github.com/JAEarly/latextable for documentation on makeing table
table = Texttable()
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['a',
                      'a',
                      'a',
                      'a',
                      'a',
                      'a'
                      ])  # automatic
table.set_cols_align(["l", "r", "r", "r", "r", "r"])
table.add_rows(table_rows)
print(table.draw() + "\n")
print('\nTexttable Table:')
print(table.draw())
print(latextable.draw_latex(table, caption="Estiation of transitivy in the Nyakatoke ",
                            label="table:another_table") + "\n")
