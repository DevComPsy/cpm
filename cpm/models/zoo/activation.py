import numpy as np
import numpy as np

class LinearActivation:

    def __init__(self, input = None, weights = None, **kwargs):
        self.input = input
        self.weights = weights
    
    def compute(self):
        """ Linear activation function
        """
        self.weights = np.asarray(self.weights * self.input)
        return None

    def config(self):
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

    def config(self):
        config = {
            'name': 'Sigmoid',
            'type': 'activation',
        }
        return config