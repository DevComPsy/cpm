import numpy as np

__all__ = ["SigmoidActivation", "CompetitiveGating"]


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


class CompetitiveGating:
    """
    A competitive attentional gating function, an attentional activation function, that incorporates stimulus salience in addition to the stimulus vector to modulate the weights.
    It formalises the hypothesis that each stimulus has an underlying salience that competes to captures attentional focus (Paskewitz and Jones, 2020; Kruschke, 2001).

    Parameters
    ----------
    input : array_like
        The input value. The stimulus representation (vector).
    values : array_like
        The values. A 2D array of values, where each row represents an outcome and each column represents a single stimulus.
    salience : array_like
        The salience value. A 1D array of salience values, where each value represents the salience of a single stimulus.
    P : float
        The power value, also called attentional normalisation or brutality, which influences the degree of attentional competition.

    Attributes
    ----------
    input : array_like
        The input value. The stimulus representation (vector).
    values : array_like
        The values value. A 2D array of values, where each row represents an outcome and each column represents a single stimulus.
    salience : array_like
        The salience value. A 1D array of salience values, where each value represents the salience of a single stimulus.
    P : float
        The power value, also called attentional normalisation or brutality. It influences the degree of attentional competition.
    gain : array_like
        The normalised attentional gain for each stimulus, corresponding to the input vector.

    Examples
    --------
    >>> input = np.array([1, 1, 0])
    >>> values = np.array([[0.1, 0.9, 0.8], [0.6, 0.2, 0.1]])
    >>> salience = np.array([0.1, 0.2, 0.3])
    >>> att = CompetitiveGating(input, values, salience, P = 1)
    >>> att.compute()
    array([[0.03333333, 0.6       , 0.        ],
           [0.2       , 0.13333333, 0.        ]])

    References
    ----------
    Kruschke, J. K. (2001). Toward a unified model of attention in associative learning. Journal of Mathematical Psychology, 45(6), 812-863.

    Paskewitz, S., & Jones, M. (2020). Dissecting exit. Journal of mathematical psychology, 97, 102371.
    """

    def __init__(self, input=None, values=None, salience=None, P=1, **kwargs):
        self.input = input
        self.values = values.copy()
        self.salience = salience.copy()
        self.P = P
        self.gain = []

    def compute(self):
        """
        Compute the activations mediated by underlying salience.

        Returns
        -------
        array_like
            The values updated with the attentional gain and stimulus vector.
        """
        self.gain = self.input * self.salience
        self.gain = self.gain**self.P
        self.gain = self.gain / np.sum(self.gain) ** (1 / self.P)
        for i in range(self.values.shape[0]):
            for k in range(self.values.shape[1]):
                self.values[i, k] = self.values[i, k] * self.gain[k]
        return self.values

    def __call__(self):
        return self.compute()

    def __repr__(self):
        return f"CompetitiveGating(input={self.input}, values={self.values}, salience={self.salience}, P={self.P})"

    def __str__(self):
        return f"CompetitiveGating(input={self.input}, values={self.values}, salience={self.salience}, P={self.P})"

    def config(self):
        """
        Print the configuration of the attentional gating function.

        Returns
        -------
        dict
            The configuration of the attentional gating function.
        """
        return {
            "input": self.input,
            "values": self.values,
            "salience": self.salience,
            "P": self.P,
            "gain": self.gain,
            "type": "Activation",
            "name": self.__class__.__name__,
        }
