import numpy as np

class Softmax:
    """
    Softmax class represents a decision-making algorithm that is often called the softmax function.
    It computes the probabilities of different choices based on the given activations and temperature.

    Attributes:
        temperature (float): The temperature parameter that controls the randomness of the decision-making process.
        activations (ndarray): The activations assigned to each choice.
        bounds (list): The lower and upper bounds of the decision probabilities.
        policies (ndarray): The computed decision policies.

    Methods:
        compute(): Computes the decision probabilities using the softmax function.
        config(): Returns the configuration of the Softmax algorithm.

    References:
        - Luce's choice axiom
        - Gibbs, J. W. (1902). Elementary principles in statistical mechanics.
    """

    def __init__(self, temperature=None, activations=None, **kwargs):
        self.temperature = temperature
        if activations is not None:
            self.activations = activations.copy()
        else:
            self.activations = np.zeros(1)
        self.policies = []

    def compute(self):
            """
            Compute the policies based on the activations and temperature.

            Returns:
                output (numpy.ndarray): Array of computed policies.
            """
            output = np.zeros(self.activations.shape[0])
            for i in range(self.activations.shape[0]):
                output[i] = np.sum(np.exp(self.activations[i] * self.temperature)) / np.sum(np.exp(self.activations * self.temperature))
            self.policies = output
            return output

    def config(self):
        config = {
            'temperature': self.temperature,
            'activations': self.activations,
            'name': self.__class__.__name__,
            'type': 'decision'
        }
        return config


class Sigmoid:
    """
    A class representing a sigmoid function.

    Attributes:
        temperature (float): The temperature parameter for the sigmoid function.
        activations (ndarray): An array of activations for the sigmoid function.
        bounds (list): A list representing the lower and upper bounds of the temperature parameter.
        policies (list): A list of output policies.

    Methods:
        compute(): Computes the sigmoid function.
        config(): Returns the configuration of the sigmoid function.
    """

    def  __init__(self, temperature=None, activations=None, **kwargs):
        self.temperature = temperature
        self.activations = activations
        self.bounds = [0, 1]
        self.policies = []

    def compute(self):
        """
        Sigmoid function that takes an n by m array of activations and returns an n
        array of outputs, where n is the number of output and m is the number of
        inputs.

        The sigmoid function is defined as: 1 / (1 + e^(-x * temperature)).

        Returns:
            ndarray: An array of outputs computed using the sigmoid function.
        """
        output = np.zeros(self.activations.shape[0])
        for i in range(self.activations.shape[0]):
            output[i] = 1 / (1 + np.exp(-self.activations[i] * self.temperature))
        self.policies = output
        return output

    def config(self):
        """
        Returns the configuration of the sigmoid function.

        Returns:
            dict: A dictionary containing the configuration of the sigmoid function.
        """
        config = {
            'temperature': self.temperature,
            'activations': self.activations,
            'name': self.__class__.__name__,
            'bounds': self.bounds,
            'type': 'decision'
        }
        return config
