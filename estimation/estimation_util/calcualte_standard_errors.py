import numpy as np

from estimation.estimation_util.calculate_hession_numerically import calculate_hessian_numerically


def calculate_standard_errors(function_value_and_score, params):

    hessian = calculate_hessian_numerically(function_value_and_score, params)
    inverse_hessian = np.linalg.inv(hessian)


    condition_number = np.linalg.cond(inverse_hessian)
    print("the condition number is: "+ str(condition_number))

    standard_errors = []
    for i in range(params.shape[0]):
        standard_errors.append(np.sqrt(inverse_hessian[i,i]))

    return standard_errors, inverse_hessian