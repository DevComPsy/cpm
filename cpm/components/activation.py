import numpy as np

class LinearActivation:
    """
    Represents a linear activation function.

    Attributes
    ----------
    input : array_like
        The input value.
    weights : array_like
        The weights value.

    Methods
    -------
    __init__(self, input=None, weights=None, **kwargs)
        Initialize the LinearActivation object.
    compute(self)
        Compute the linear activation function.
    config(self)
        Get the configuration of the linear activation function.
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
            'name': 'Linear',
            'type': 'activation',
        }
        return config

# The SigmoidActivation class implements the sigmoid activation function for
# neural networks.
class SigmoidActivation:
    """
    Represents a sigmoid activation function.

    Attributes
    ----------
    input : numpy.ndarray
        The input values.
    weights : numpy.ndarray
        The weights associated with the input values.

    Methods
    -------
    compute()
        Compute the activation value using the sigmoid function.
    config()
        Get the configuration of the activation function.
    """

    def __init__(self, input=None, weights=None, **kwargs):
        """
        Initialize the SigmoidActivation object.

        Parameters
        ----------
        input : numpy.ndarray, optional
            The input values (default is None).
        weights : numpy.ndarray, optional
            The weights associated with the input values (default is None).
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
            'name': 'Sigmoid',
            'type': 'activation',
        }
        return config