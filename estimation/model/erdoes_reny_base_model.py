
import numpy as np

"""
A simple model for nyakatoke

Only one parameter, which determines the probability for agents to form links

"""

class ErdoesRenyModel():

    def __init__(self, n):
        self.mn = n*(n-1)  # no self loops allowed, therefore n*(n-1) decisions

    def get_initial_params(self):
        return np.array([0.4,-0.1]) # initially gamma=1, and a=0 -> corresponds to 0.5 density
    """
    Implementation similar like in AI models:
        : forward pass calculates the mue, from the original parameters
        : backwards pass calculates the derivative on the parameters form the derivatife on the mue
        
    For this simple model we have
        mue_i,j = a   for all i,j
    
    """
    def forward(self, a):
        a_scalar = a[0]
        return np.array([a_scalar]*self.mn)

    def backward(self, derivative_mue):
        return np.array([np.sum(derivative_mue)])














