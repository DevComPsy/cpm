import numpy as np
import numpy as np

# The above class represents a linear activation function in a neural network.
import numpy as np

import numpy as np

class LinearActivation:
    """
    Represents a linear activation function.
    """

    def __init__(self, input=None, weights=None, **kwargs):
        """
        Initialize the LinearActivation object.

        Parameters
        ----------
        input : array_like
            The input value.
        weights : array_like
            The weights value.
        **kwargs : 
            Additional keyword arguments.

        Returns
        -------
            None
        """
        self.input = input
        self.weights = weights
    
    def compute(self):
        """
        Compute the linear activation function.

        Returns:
            None
        """
        self.weights = np.asarray(self.weights * self.input)
        return None

    def config(self):
        """
        Get the configuration of the linear activation function.

        Returns:
            dict: The configuration dictionary.
        """
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