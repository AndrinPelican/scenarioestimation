import numpy as np

"""
A Linear Model, for the standard regression equation

"""


class LinearModel():

    def __init__(self, X):
        """
        :param X:  N x anz_parama Covariate matrix
        """
        self.X = X
        self.N = X.shape[0]
        self.anz_params = X.shape[1]
        self.initial_gamma = 0.02
        self.initial_params = [0]*self.anz_params

    def get_initial_params(self):
        initial_params_list =[self.initial_gamma]+ list(self.initial_params)
        return np.array(initial_params_list)  # ititially gamma=0.5, and a=0 -> corresponds to 0.5 density

    """
    Implementation similar like in AI models:
        : forward pass calculates the mue, from the orignal parameters
        : backwards pass calculates the derivative on the parameters form the derivatife on the mue
        
    For this simple model we have
        mue_i,j = a   for all i,j
        
    """

    def forward(self, a):
        return np.matmul(self.X, a)

    def backward(self, derivative_mue):
        return np.matmul(np.transpose(self.X,[1,0]), derivative_mue)

    """
    Mock backward to 0
    
    This is to see what the impact of over fitting is.
    substituting backwards to 0 results in no change in the parameters, 
    
    """

