import numpy as np


class LinearActivation:
    """
    Represents a linear activation function.
    This function simply multiplies the stimulus (input) vector with the values (weights) vector
    without any additional scaling.

    Attributes
    ----------
    input : array_like
        The input value. The stimulus representation (vector).
    weights : array_like
        The weights value. A 2D array of weights, where each row represents an outcome and each column represents a single stimulus.

    """

    def __init__(self, input=None, weights=None, **kwargs):
        """
        Initialize the LinearActivation object.

        Parameters
        ----------
        input : array_like
            The input value. The stimulus representation (vector).
        weights : array_like
            The weights value. A 2D array of weights, where each row represents an outcome and each column represents a single stimulus.
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
        It simply multiplies the nominal stimulus representation as input with the weights.

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
            "name": "Linear",
            "type": "activation",
        }
        return config


# The SigmoidActivation class implements the sigmoid activation function for
# neural networks.
class SigmoidActivation:
    """
    Represents a sigmoid activation function.


    Attributes
    ----------
        input : array_like
            The input value. The stimulus representation (vector).
        weights : array_like
            The weights value. A 2D array of weights, where each row represents an outcome and each column represents a single stimulus.

    """

    def __init__(self, input=None, weights=None, **kwargs):
        """
        Initialize the SigmoidActivation object.

        Parameters
        ----------
        input : array_like
            The input value. The stimulus representation (vector).
        weights : array_like
            The weights value. A 2D array of weights, where each row represents an outcome and each column represents a single stimulus.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.input = input
        self.weights = weights

    def compute(self):
        """
        Compute the activation value using the sigmoid function.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        numpy.ndarray
            The computed activation value.
        """
        return np.asarray(1 / (1 + np.exp(-self.input * self.weights)))

    def config(self):
        """
        Get the configuration of the activation function.

        Returns
        -------
        dict
            The configuration of the activation function.
        """
        config = {
            "name": "Sigmoid",
            "type": "activation",
        }
        return config
