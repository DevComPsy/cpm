import numpy as np

__all__ = ["Softmax", "Sigmoid", "GreedyRule", "ChoiceKernel"]


class Softmax:
    """
    Softmax class for computing policies based on activations and temperature.

        The softmax function is defined as: e^(temperature * x) / sum(e^(temperature * x)).

    Attributes
    ----------
    temperature : float
        The inverse temperature parameter for the softmax computation.
    activations : numpy.ndarray
        Array of activations for each possible outcome/action. It should be
        a 2D ndarray, where each row represents an outcome and each column
        represents a single stimulus.
    policies : numpy.ndarray
        Array of computed policies.
    shape : tuple
        The shape of the activations array.
    """

    def __init__(self, temperature=None, activations=None, **kwargs):
        """
        Parameters
        ----------
        temperature : float
            The inverse temperature parameter for the softmax computation.
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

    def __repr__(self):
        return f"{self.__class__.__name__}(temperature={self.temperature}, activations={self.activations})"

    def __str__(self):
        return f"{self.__class__.__name__}(temperature={self.temperature}, activations={self.activations})"

    def __call__(self):
        return self.compute()


class Sigmoid:
    """
    A class representing a sigmoid function that takes an n by m array of activations and returns an n
    array of outputs, where n is the number of output and m is the number of
    inputs.

        The sigmoid function is defined as: 1 / (1 + e^(-temperature * (x - beta))).

    Parameters
    ----------
    temperature : float
        The inverse temperature parameter for the sigmoid function.
    beta : float
        It is the value of the output activation that results in an output rating
        of P = 0.5.
    activations : ndarray
        An array of activations for the sigmoid function.

    Attributes
    ----------
    temperature : float
        The inverse temperature parameter for the sigmoid function.
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
        output: ndarray
            A 2D array of outputs computed using the sigmoid function.
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
            "type": "decision",
        }
        return config

    def __repr__(self):
        return f"{self.__class__.__name__}(temperature={self.temperature}, activations={self.activations}, beta={self.beta})"

    def __str__(self):
        return f"{self.__class__.__name__}(temperature={self.temperature}, activations={self.activations}, beta={self.beta})"

    def __call__(self):
        return self.compute()


class GreedyRule:
    """
    A class representing an ε-greedy rule based on Daw et al. (2006).

    Parameters
    ----------
    activations : ndarray
        An array of activations for the greedy rule.
    epsilon : float
        Exploration parameter. The probability of selecting a random action.

    Attributes
    ----------
    activations : ndarray
        An array of activations for the greedy rule.
    epsilon : float
        Exploration parameter. The probability of selecting a random action.
    policies : ndarray
        An array of outputs computed using the greedy rule.
    shape : tuple
        The shape of the activations array.

    References
    ----------
    Daw, N. D., O’Doherty, J. P., Dayan, P., Seymour, B., & Dolan, R. J. (2006). Cortical substrates for exploratory decisions in humans. Nature, 441(7095), Article 7095. https://doi.org/10.1038/nature04766
    """

    def __init__(self, activations=None, epsilon=0, **kwargs):
        self.activations = np.asarray(activations.copy())
        self.epsilon = epsilon
        self.policies = []
        self.shape = self.activations.shape
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])
        self.run = False

    def compute(self):
        """
        Computes the greedy rule.

        Returns
        -------
        output: ndarray
            A 2D array of outputs computed using the greedy rule.
        """
        output = self.activations.sum(axis=1)
        policies = np.zeros(output.shape)
        maximum = np.max(output)
        policies[output != maximum] = self.epsilon * 1
        policies[output == maximum] = 1 - (output.shape[0] - 1) * self.epsilon
        policies[output <= 0] = 0
        if np.all(policies == 0):
            policies.fill(1 / policies.shape[0])
        else:
            policies = policies / policies.sum()  # normalise
        self.policies = policies
        self.run = True
        return self.policies

    def choice(self):
        """
        Chooses the action based on the greedy rule.

        Returns
        -------
        action: int
            The chosen action based on the greedy rule.
        """
        if not self.run:
            self.compute()
        out = np.random.choice(self.shape[0], p=self.policies)
        return out

    def config(self):
        """
        Returns the configuration of the greedy rule.

        Returns
        -------
        config: dict
            A dictionary containing the configuration of the greedy rule.

            - activations (ndarray): An array of activations for the greedy rule.
            - name (str): The name of the greedy rule.
            - type (str): The class of function it belongs.
        """
        config = {
            "activations": self.activations,
            "name": self.__class__.__name__,
            "type": "decision",
        }
        return config

    def __repr__(self):
        return f"{self.__class__.__name__}(activations={self.activations}, epsilon={self.epsilon})"

    def __str__(self):
        return f"{self.__class__.__name__}(activations={self.activations}, epsilon={self.epsilon})"

    def __call__(self):
        return self.compute()


class ChoiceKernel:
    """
    A class representing a choice kernel based on a softmax function that incorporates the frequency of choosing an action.
    It is based on Equation 7 in Wilson and Collins (2019).

    Parameters
    ----------
    temperature_activations : float
        The inverse temperature parameter for the softmax computation.
    temperature_kernel : float
        The inverse temperature parameter for the kernel computation.
    activations : ndarray, optional
        An array of activations for the softmax function.
    kernel : ndarray, optional
        An array of kernel values for the softmax function.

    Attributes
    ----------
    temperature_a : float
        The inverse temperature parameter for the softmax computation.
    temperature_k : float
        The inverse temperature parameter for the kernel computation.
    activations : ndarray
        An array of activations for the softmax function.
    kernel : ndarray
        An array of kernel values for the softmax function.
    policies : ndarray
        An array of outputs computed using the softmax function.
    shape : tuple
        The shape of the activations array.

    Notes
    -----

    In order to get Equation 6 from Wilson and Collins (2019), either set `activations` to None (default) or set it to 0.

    See Also
    --------
    [cpm.components.learning.KernelUpdate](cpm.components.learning.KernelUpdate): A class representing a kernel update (Equation 5; Wilson and Collins, 2019) that updates the kernel values.

    References
    ----------
    Wilson, R. C., & Collins, A. G. E. (2019). Ten simple rules for the computational modeling of behavioral data. eLife, 8, Article e49547.

    """

    def __init__(
        self,
        temperature_activations=0.5,
        temperature_kernel=0.5,
        activations=None,
        kernel=None,
        **kwargs,
    ):
        self.temperature_a = temperature_activations
        self.temperature_k = temperature_kernel
        self.activations = activations.copy()
        self.kernel = kernel.copy()
        self.policies = []
        if activations is None:
            self.activations = np.zeros(1)
        self.shape = self.kernel.shape

    def compute(self):
        output = np.zeros(self.shape[0])
        values = self.activations * self.temperature_a
        kernels = self.kernel * self.temperature_k
        # activation of output unit for action/outcome
        nominator = np.exp(np.sum(values, axis=1) * kernels)
        # denominator term for scaling
        denominator = np.sum(np.exp(np.sum(values, axis=1) * kernels))
        output = nominator / denominator
        self.policies = output
        return output

    def choice(self):
        return np.random.choice(self.shape[0], p=self.policies)

    def config(self):
        config = {
            "temperature_activations": self.temperature_a,
            "temperature_kernel": self.temperature_k,
            "activations": self.activations,
            "kernel": self.kernel,
            "name": self.__class__.__name__,
            "type": "decision",
        }
        return config

    def __repr__(self):
        return f"{self.__class__.__name__}(temperature_activations={self.temperature_a}, temperature_kernel={self.temperature_k}, activations={self.activations}, kernel={self.kernel})"

    def __str__(self):
        return f"{self.__class__.__name__}(temperature_activations={self.temperature_a}, temperature_kernel={self.temperature_k}, activations={self.activations}, kernel={self.kernel})"

    def __call__(self):
        return self.compute()


# ChoiceKernel(
#     temperature_activations=1,
#     temperature_kernel=1,
#     activations=np.array([[0.1], [0], [0.2]]),
#     kernel=np.array([0.1, 0.9, 0.1]),
# ).compute()