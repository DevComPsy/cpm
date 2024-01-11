"""
all sorts of activation functions
"""
import numpy as np


def linear(input, weights):
    """ Linear activation function
    """
    return np.asarray(weights * input)

def sigmoid(input, weights):
    """ Sigmoid activation function
    """
    return np.asarray(1 / (1 + np.exp(-input * weights)))