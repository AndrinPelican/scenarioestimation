
import numpy as np

"""
Calculate the Hessian numerically

"""

def calculate_hessian_numerically(function_value_and_score, params, epsilon = 0.00001):

    colums = []
    for i in range(params.shape[0]):
        colums.append(derivative_with_respect_to_one_variable(function_value_and_score, params, i, epsilon))
    hessian = np.stack(colums, axis=0)
    return hessian




def derivative_with_respect_to_one_variable(function_value_and_score, params, i, epsilon):

    params_plus  = np.copy(params)
    params_plus[i] = params[i] + epsilon

    params_minus = np.copy(params)
    params_minus[i] = params[i] - epsilon

    value, der_plus = function_value_and_score(params_plus)
    value, der_minus = function_value_and_score(params_minus)

    derivative = (der_plus-der_minus)/(2*epsilon)

    return derivative


