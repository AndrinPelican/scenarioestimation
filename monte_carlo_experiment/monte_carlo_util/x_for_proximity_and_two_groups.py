
import numpy as np

"""
This creates linear model with two group variables X[0]m X[1] and one variable on how close nodes are X[2] with 
range[0,1]

"""

def proximity_and_two_groups(n):

    # preparation
    X = np.zeros((n*(n-1),3))
    k = 0

    # filling it up
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i< n/2:
                X[k, 0] = 1
                X[k, 1] = 0
            else:
                X[k, 0] = 0
                X[k, 1] = 1
            X[k, 2] = abs(j-i)/n
            k += 1

    return X

