import numpy as np
import numpy as np

class LinearActivation:

    def __init__(self, input, weights, **kwargs):
        self.input = input
        self.weights = weights
    
    def compute(self):
        """ Linear activation function
        """
        return np.asarray(self.weights * self.input)

    def config():
        config = {
            'name': 'Linear',
            'type': 'activation',
        }
        return config

class SigmoidActivation:

    def compute(self, input, weights, **kwargs):
        """ Sigmoid activation function
        """
        return np.asarray(1 / (1 + np.exp(-input * weights)))

    def config():
        config = {
            'name': 'Sigmoid',
            'type': 'activation',
        }
        return config