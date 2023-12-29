
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize

def target_closed_form(x):
    "Closed form target function, for motivation see test_estimation_for_n_3_graph"
    gamma = x[0]
    a = x[1]
    F = norm.cdf
    value = - (1-F(a))**2 *\
            (1-F(a+gamma))*F(a)\
            * (F(a) ** 2 + 2 *  (F(a + gamma)-F(a)) *F(a))


    print(value)
    return value

def target_closed_form_with_log_punishment(x):

    if x[0]<=0:
        return 10000000000

    return target_closed_form(x) + - 0.00000001* np.log(x[0])


x0 = np.array([0.7, 0.5])
res = minimize(target_closed_form_with_log_punishment, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print(res.x)

