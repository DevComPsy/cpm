import numpy as np

__all__ = ['softmax', 'sigmoid']

# NOTE: references should include Luce's choice axiom and Gibbs 1902
def softmax(temperature, weights, ***args, **kwargs):
    """
    Softmax function
    """
    output = np.zeros(weights.shape[0])
    for i in range(weights.shape[0]):
        output[i] = np.sum(np.exp(weights[i] * temperature)) / np.sum(np.exp(weights * temperature))
    return output

# NOTE: references should include gluck and bower 1988
def sigmoid(temperature, activations, ***args, **kwargs):
    """
    Sigmoid function that takes an n by m array of activations and returns an n
    array of outputs, where n is the number of output and m is the number of
    inputs.

    The sigmoid function is defined as: 1 / (1 + e^(-x * temperature)).
    """
    output = np.zeros(activations.shape[0])
    for i in range(activations.shape[0]):
        output[i] = 1 / (1 + np.exp(-activations[i] * temperature))
    return output
