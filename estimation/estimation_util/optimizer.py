"""

We use the scipy optimize package to optimize our functions:

This is a minimal example:

import numpy as np
from scipy.optimize import minimize

def rosen(x):
    "The Rosenbrock function"
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
es = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der

res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
               options={'disp': True})

"""

from scipy.optimize import minimize




def estimate_parameters(cost_function, x0):

    def cost(x):
        value, derivative = cost_function(x)
        return value

    # jac stands for jacobian = derivative matrix
    def jac(x):
        value, derivative = cost_function(x)
        # print("gradient")
        # print(derivative)
        return derivative

    # bounds are only possible with: e L-BFGS-B, TNC, COBYLA or SLSQP
    # L-BFGS-B seems at least to do
    res = minimize(cost, x0, method='BFGS', jac=jac, options={'disp': True})

    # returning the optimal parameters
    return res.x, res.fun


def estimate_parameters_no_grdient(cost_function, x0):

    def cost(x):
        value, derivative = cost_function(x)
        return value


    # bounds are only possible with: e L-BFGS-B, TNC, COBYLA or SLSQP
    # L-BFGS-B seems at least to do
    res = minimize(cost, x0, method='Nelder-Mead', options={'disp': True})

    # returning the optimal parameters
    return res.x, res.fun



