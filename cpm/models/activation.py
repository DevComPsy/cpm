import numpy as np

__all__ = ["SigmoidActivation", "CompetitiveGating", "ProspectUtility", "Offset"]


class SigmoidActivation:
    """
    Represents a sigmoid activation function.

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

        Returns
        -------
        numpy.ndarray
            The computed activation value.
        """
        return np.asarray(1 / (1 + np.exp(-self.input * self.weights)))


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


class ProspectUtility:
    """
    A class for computing choice utilities based on prospect theory.

    Parameters
    ----------
    magnitudes : numpy.ndarray
        A nested array where the outer dimension represents trials, with each trial
        containing the potential outcome magnitudes for each option.
    probabilities : numpy.ndarray
        A nested array (with the same shape as magnitudes) where each entry contains
        the probability of the corresponding outcome.
    alpha : float
        The utility curvature parameter, used for both gains and losses.
    beta : float, optional
        An optional parameter for the utility function used by Tversky and Kahneman (1992) for losses, defaults to `alpha` if not provided.
    lambda_loss : float
        The loss aversion parameter (scaling losses relative to gains).
    gamma : float
        The probability weighting curvature parameter (for gains with "tk" and both gains and losses with "power").
    delta : float, optional
        The attractiveness parameter, which determines the elevation of the weighting function in `prelec` and `gw` weighting functions. In `tk`, it is the probability weighting for losses. Defaults to `gamma` if not provided.
    utility_curve : callable, optional
        An optional utility function that takes the magnitude, alpha, and lambda_loss, and returns the utilities of each choice options. The default is a power utility function, see Notes.
    weighting : str
        The definition of the weighting function. Should be one of 'tk', 'pd', or 'gw'. See Notes for details.
    **kwargs : dict, optional
        Additional keyword arguments.

    Notes
    -----

    The different weighting functions currently implemented are:

        - `tk`: Tversky & Kahneman (1992).
        - `prelec`: Prelec (1998).
        - `gw`: Gonzalez & Wu (1999).
        - `power` : Simple power function: w(p) = p^gamma
        

    Following Tversky & Kahneman (1992), the expected utility U of a choice option is defined as:

    $$
    \\mathcal{U} = \\sum_{i=1}^{n} w(p_i) \\cdot u(x_i)
    $$

    where $w$ is a weighting function of the probability p of a potential outcome,
    and $u$ is the utility function of the magnitude x of a potential outcome.
    The utility function $u$ is defined as a power function for both gains and losses. It is implemented
    after Equation 5 in Tversky & Kahneman (1992):

    $$
    u(x) =
    \\begin{cases}
        x^\\alpha & \\text{if } x \\geq 0 \\\\
        -\\lambda \\cdot (-x)^\\alpha & \\text{if } x < 0
    \\end{cases}
    $$

    where $\\alpha$ is the utility curvature parameter, and $\\lambda$ is the loss aversion parameter.
    The weighting function is implemented after Equation 6 in Tversky & Kahneman (1992):

    $$
    w(p) = \\frac{p^\\gamma}{(p^\\gamma + (1 - p)^\\gamma)^{1/\\gamma}}
    $$

    where `gamma`, denoted via $\\gamma$, is the discriminability parameter of the weighting function.
    In the original formulation of Tversky & Kahneman (1992), losses are weighted with a different parameter,
    `delta`, denoted via $\\delta$, that replaces $\\gamma$ in the weighting function for losses.
    In the current implementation, whether it is a gain or less is determined by the sign of the corresponding
    magnitude.

    Several other definitions of the weighting function have been proposed in the literature,
    most notably in Prelec (1998) and Gonzalez & Wu (1999).
    Prelec (equation 3.2, 1998, pp. 503) proposed the following definition:

    $$
    w(p) = \\exp(-\\delta \\cdot (-\\log(p))^\\gamma)
    $$

    where `delta`, $\\delta$, and `gamma`, $\\gamma$, are the attractiveness and discriminability parameters of the weighting function.
    Gonzalez & Wu (equation 3, 1999, pp. 139) proposed the following definition:

    $$
    w(p) = \\frac{\\delta \\cdot p^\\gamma}{\\delta \\cdot p^\\gamma + (1 - p)^\\gamma}
    $$

    Examples
    --------
    >>> from cpm.models.activations import ProspectUtility
    >>> magnitudes = [[5, 0], [10, -10]]
    >>> probabilities = [[0.8, 0.2], [0.5, 0.5]]
    >>> model = ProspectUtility(
            magnitudes=magnitudes,
            probabilities=probabilities,
            alpha=0.88,
            lambda_loss=2.25,
            gamma=0.61,
            delta=1.0,
            weighting="tk"
        )
    >>> expected_utilities = model.compute()
    >>> print(expected_utilities)

    References
    ----------
    Gonzalez, R., & Wu, G. (1999). On the shape of the probability weighting function. Cognitive psychology, 38(1), 129-166.

    Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263–291.

    Prelec, D. (1998). The probability weighting function. Econometrica, 497-527.

    Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and uncertainty, 5, 297-323.
    """

    def __init__(
        self,
        magnitudes=None,
        probabilities=None,
        alpha=1,
        beta=None,
        lambda_loss=1,
        gamma=1,
        delta=1,
        utility_curve=None,
        weighting="tk",
        **kwargs,
    ):
        self.magnitudes = np.asarray(magnitudes.copy())
        self.magnitudes = np.array(
            [np.array(self.magnitudes[i], dtype=float) for i in range(self.magnitudes.shape[0])],
            dtype=object,
        )
        self.probabilities = np.asarray(probabilities.copy())
        self.probabilities = np.array(
            [np.array(self.probabilities[i], dtype=float) for i in range(self.probabilities.shape[0])],
            dtype=object,
        )
        self.alpha = alpha
        if beta is not None:
            self.beta = beta
        else:
            self.beta = alpha
        self.lambda_loss = lambda_loss
        self.gamma = gamma
        if delta is not None:
            self.delta = delta
        else:
            self.delta = gamma

        self.shape = self.magnitudes.shape
        if self.shape != self.probabilities.shape:
            raise ValueError("magnitudes and probabilities do not have the same shape.")

        # Choose weighting function based on the argument
        if weighting == "tk":
            self.__weighting_fun = self.__weighting_tk
        elif weighting == "power":
            self.__weighting_fun = self.__weighting_power
        elif weighting == "prelec":
            self.__weighting_fun = self.__weighting_prelec
        elif weighting == "gw":
            self.__weighting_fun = self.__weighting_gw
        else:
            raise ValueError("Invalid weighting type. Must be one of: 'tk', 'power', 'prelec', 'gw'.")

        if utility_curve is None:
            self.__utility_curve = self.__utility_power
        elif callable(utility_curve):
            self.__utility_curve = utility_curve
        else:
            raise ValueError("Utility curve must be a callable function.")
            
        self.utilities = []
        self.weights = []
        self.expected_utility = []
        self.weighting = weighting

    def __utility_power(self, x=None, alpha=None, beta=None, lambda_loss=None):
        # For gains: x^alpha; for losses: -lambda_loss * |x|^alpha
        # Ensure x is not None and is a numpy array
        if x is None:
            raise ValueError("Magnitudes cannot be None.")
        x = np.asarray(x, dtype=float)
        expected = np.zeros_like(x, dtype=float)
        gains = x >= 0
        losses = ~gains
        expected[gains] = np.power(x[gains], self.alpha)
        expected[losses] = -self.lambda_loss * np.power(-x[losses], self.beta)
        return expected

    def __weighting_tk(self, x=None, magnitudes=None):
        # Vectorized implementation for speed
        powers = np.where(np.asarray(magnitudes) > 0, self.gamma, self.delta)
        # x is an array of probabilities, powers is an array of exponents
        numerator = np.power(x, powers)
        denominator = np.power(numerator + np.power(1 - x, powers), 1 / powers)
        return numerator / denominator

    def __weighting_power(self, x=None, magnitudes=None):
        return np.power(x, self.gamma)

    def __weighting_prelec(self, x=None, magnitudes=None):
        return np.exp(-self.delta * np.power(-np.log(x), self.gamma))

    def __weighting_gw(self, x=None, magnitudes=None):
        numerator = self.delta * np.power(x, self.gamma)
        denominator = numerator + np.power(1 - x, self.gamma)
        return numerator / denominator

    def weight_probability(self, x, **kwargs):
        return self.__weighting_fun(x, **kwargs)

    def compute(self):
        """
        Compute the expected utility of each choice option.

        Returns
        -------
        numpy.ndarray
            The computed expected utility of each choice option.
        """
        # Determine the utilities of the potential outcomes, for each choice option and each trial.
        self.utilities = np.array(
            [self.__utility_curve(x=self.magnitudes[j], alpha=self.alpha, lambda_loss=self.lambda_loss) for j in range(self.shape[0])],
            dtype=object,
        )
        # Determine the weights of the potential outcomes, for each choice option and each trial.
        self.weights = np.array(
            [self.__weighting_fun(x=self.probabilities[j], magnitudes=self.magnitudes[j]) for j in range(self.shape[0])],
            dtype=object,
        )
        # Determine the expected utility of each choice option for each trial.
        self.expected_utility = np.array(
            [np.sum(self.weights[j] * self.utilities[j]) for j in range(self.shape[0])],
        )
        return self.expected_utility

    def __call__(self):
        return self.compute()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"magnitudes={self.magnitudes}, probabilities={self.probabilities}, "
            f"alpha={self.alpha}, lambda_loss={self.lambda_loss}, gamma={self.gamma}, "
            f"delta={self.delta}, weighting={self.weighting}"
            f")"
        )

    def __str__(self):
        return self.__repr__()

class Offset:
    """
    A class for adding a scalar to one element of an input array.
    In practice, this can be used to "shift" or "offset" the "value" of one particular stimulus, for example to represent a consistent bias for (or against) that stimulus.

    Parameters
    ----------
    input : array_like
        The input value. The stimulus representation (vector).
    offset : float
        The value to be added to one element of the input.
    index : int
        The index of the element of the input vector to which the offset should be added.
    **kwargs : dict, optional
        Additional keyword arguments.


    Examples
    --------
    >>> vals = np.array([2.1, 1.1])
    >>> offsetter = Offset(input = vals, offset = 1.33, index = 0)
    >>> offsetter.compute()
    array([3.43, 1.1])
    """

    def __init__(self, input=None, offset=0, index=0, **kwargs):
        self.input = np.asarray(input.copy())
        self.offset = offset
        self.index = index
        self.output = self.input.copy()

    def compute(self):
        """
        Add the offset to the requested input element.

        Returns
        -------
        numpy.ndarray
            The stimulus representation (vector) with offset added to the requested element.
        """
        self.output[self.index] += self.offset
        return self.output

    def __call__(self):
        return self.compute()

    def __repr__(self):
        return f"{self.__class__.__name__}(input={self.input}, offset={self.offset}, index={self.index})"

    def __str__(self):
        return f"{self.__class__.__name__}(input={self.input}, offset={self.offset}, index={self.index})"
