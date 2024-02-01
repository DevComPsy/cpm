import numpy as np

import numpy as np


class Softmax:
    """
    Softmax class for computing policies based on activations and temperature.

        The softmax function is defined as: e^(temperature * x) / sum(e^(temperature * x)).
    """

    def __init__(self, temperature=None, activations=None, **kwargs):
        """
        Parameters
        ----------
        temperature : float
            The temperature parameter for softmax computation.
        activations : numpy.ndarray
            Array of activations for each possible outcome/action. It should be
            a 2D ndarray, where each row represents an outcome and each column
            represents a single stimulus.

        Examples
        --------
        >>> from cpm.components.decision import Softmax
        >>> import numpy as np
        >>> temperature = 1
        >>> activations = np.array([[0.1, 0, 0.2], [-0.6, 0, 0.9]])
        >>> softmax = Softmax(temperature=temperature, activations=activations)
        >>> softmax.compute()
        array([0.45352133, 0.54647867])

        >>> softmax.config()
        {
            "temperature"   : 1,
            "activations":
                array([[ 0.1,  0. ,  0.2],
                [-0.6,  0. ,  0.9]]),
            "name"  : "Softmax",
            "type"  : "decision",
        }
        >>> Softmax(temperature=temperature, activations=activations).compute()
        array([0.45352133, 0.54647867])
        """
        self.temperature = temperature
        if activations is not None:
            self.activations = activations.copy()
        else:
            self.activations = np.zeros(1)
        self.policies = []
        self.shape = self.activations.shape
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])

    def compute(self):
        """
        Compute the policies based on the activations and temperature.

        Returns
        -------
        output (numpy.ndarray): Array of computed policies.
        """
        output = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            output[i] = np.sum(np.exp(self.activations[i] * self.temperature)) / np.sum(
                np.exp(self.activations * self.temperature)
            )
        self.policies = output
        return output

    def config(self):
        """
        Get the configuration of the Softmax class.

        Returns
        ------
        config (dict): Dictionary containing the configuration parameters.
        """
        config = {
            "temperature": self.temperature,
            "activations": self.activations,
            "name": self.__class__.__name__,
            "type": "decision",
        }
        return config


class Sigmoid:
    """
    A class representing a sigmoid function that takes an n by m array of activations and returns an n
    array of outputs, where n is the number of output and m is the number of
    inputs.

        The sigmoid function is defined as: 1 / (1 + e^(-temperature * (x - beta))).

    Parameters
    ----------
    temperature : float
        The temperature parameter for the sigmoid function.
    beta : float
        It is the value of the output activation that results in an output rating
        of P = 0.5.
    activations : ndarray
        An array of activations for the sigmoid function.

    Attributes
    ----------
    temperature : float
        The temperature parameter for the sigmoid function.
    beta : float
        The bias parameter for the sigmoid function. It is the value of the
        output activation that results in an output rating of P = 0.5.
    activations : ndarray
        An array of activations for the sigmoid function.
    policies : ndarray
        An array of outputs computed using the sigmoid function.
    shape : tuple
        The shape of the activations array.

    """

    def __init__(self, temperature=None, activations=None, beta=0, **kwargs):
        self.temperature = temperature
        self.beta = beta
        self.activations = np.asarray(activations.copy())
        self.policies = []
        self.shape = self.activations.shape
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])

    def compute(self):
        """
        Computes the Sigmoid function.

        Returns
        -------
            ndarray: An array of outputs computed using the sigmoid function.
        """
        output = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            output[i] = 1 / (
                1
                + np.exp((np.sum(self.activations[i]) - self.beta) * -self.temperature)
            )
        self.policies = output
        return output

    def config(self):
        """
        Returns the configuration of the sigmoid function.

        Returns
        -------
        config: dict
            A dictionary containing the configuration of the sigmoid function.

            - temperature (float): The temperature parameter for the sigmoid function.
            - beta (float): The bias parameter for the sigmoid function.
            - activations (ndarray): An array of activations for the sigmoid function.
            - name (str): The name of the sigmoid function.
            - type (str): The class of function it belongs.
        """
        config = {
            "temperature": self.temperature,
            "activations": self.activations,
            "name": self.__class__.__name__,
            "bounds": self.bounds,
            "type": "decision",
        }
        return config
