import numpy as np

__all__ = ["SigmoidActivation", "CompetitiveGating", "ProspectUtility"]


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

class ProspectUtility:
    """
    A class for computing choice utilities based on prospect theory.
    Following Tversky & Kahneman (1992), the expected utility U of a choice option is defined as:

        U = sum(w(p) * u(x)),

    where w is a weighting function of the probability p of a potential outcome,
    and u is the utility function of the value x of a potential outcome.
    These functions are defined as follows (equations 6 and 5 respectively in Tversky & Kahneman, 1992, pp. 309):

        w(p) = p^beta / (p^beta + (1 - p)^beta)^(1/beta),\n
        u(x) = ifelse(x >= 0, x^alpha_pos, -lambda * (-x)^alpha_neg),

    where beta is the discriminability parameter of the weighting function;
    alpha_pos and alpha_neg are the risk attitude parameters in the gain and loss domains respectively,
    and lambda is the loss aversion parameter.

    Several other definitions of the weighting function have been proposed in the literature,
    most notably in Prelec (1998) and Gonzalez & Wu (1999).
    Prelec (equation 3.2, 1999, pp. 503) proposed the following definition:

        w(p) = exp(-delta * (-log(p))^beta),
    
    where delta and beta are the attractiveness and discriminability parameters of the weighting function.
    Gonzalez & Wu (equation 3, 1999, pp. 139) proposed the following definition:

        w(p) = (delta * p^beta) / ((delta * p^beta) + (1-p)^beta).

    Parameters
    ----------
    outcomes : array_like
        The values of potential outcomes for each choice option.
        Should be a nested array where the outer dimension represents trials,
        followed by options within each trial, followed by potential outcomes within each option.
    probabilities : array_like
        The probabilities of potential outcomes for each choice option.
        Should be a nested array where the outer dimension represents trials,
        followed by options within each trial, followed by potential outcomes within each option.
    alpha_pos : float
        The risk attitude parameter for non-negative outcomes, which determines the curvature of the utility function in the gain domain.
        If alpha_neg is undefined, alpha_pos will be used for both the gain and loss domains.
    alpha_neg : float
        The risk attitude parameter for negative outcomes, which determines the curvature of the utility function in the loss domain.
    lambda : float
        The loss aversion parameter, which scales the utility of negative outcomes relative to non-negative outcomes.
    beta : float
        The discriminability parameter, which determines the curvature of the weighting function.
    delta : float
        The attractiveness parameter, which determines the elevation of the weighting function.
    w_fun : str
        The definition of the weighting function. Should be one of 'tversky_kahneman', 'prelec', or 'gonzalez_wu'.
    **kwargs : dict, optional
        Additional keyword arguments.

    Attributes
    ----------
    outcomes : ndarray
        The values of potential outcomes for each choice option, for each trial.
        Should be a nested array where the outer dimension represents trials,
        followed by options within each trial, followed by potential outcomes within each option.
    probabilities : ndarray
        The probabilities of potential outcomes for each choice option, for each trial.
        Should be a nested array where the outer dimension represents trials,
        followed by options within each trial, followed by potential outcomes within each option.
    alpha_pos : float
        The risk attitude parameter for non-negative outcomes, which determines the curvature of the utility function in the gain domain.
        If alpha_neg is undefined, alpha_pos will be used for both the gain and loss domains.
    alpha_neg : float
        The risk attitude parameter for negative outcomes, which determines the curvature of the utility function in the loss domain.
    _lambda : float
        The loss aversion parameter, which scales the utility of negative outcomes relative to non-negative outcomes.
    beta : float
        The discriminability parameter, which determines the curvature of the weighting function.
    delta : float
        The attractiveness parameter, which determines the elevation of the weighting function.
    w_fun : str
        The definition of the weighting function. One of 'tversky_kahneman', 'prelec', or 'gonzalez_wu'.
    utilities : ndarray
        The computed utilities of potential outcomes for each choice option, for each trial.
    weights : ndarray
        The computed weights of potential outcomes for each choice option, for each trial.
    expected_utility : ndarray
        The computed expected utility of each choice option for each trial.

    Examples
    --------
    >>> vals = np.array([
            [np.array([1, 40]), np.array([10])],
            [np.array([1, 10]), np.array([8])]
        ], dtype=object)
    >>> probs = np.array([
            [np.array([0.95, 0.05]), np.array([1])],
            [np.array([0.6, 0.4]), np.array([1])]
        ], dtype=object)
    >>> prospect = ProspectUtility(
            outcomes=vals, probabilities=probs, alpha_pos = 0.85, beta = 0.9
        )
    >>> prospect.compute()
    array([[2.44583162, 7.07945784],
           [3.46509053, 5.85634278]])

    References
    ----------
    Gonzalez, R., & Wu, G. (1999). On the shape of the probability weighting function. Cognitive psychology, 38(1), 129-166.

    Prelec, D. (1998). The probability weighting function. Econometrica, 497-527.

    Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and uncertainty, 5, 297-323.
    """

    def __init__(self,
                 outcomes=None, probabilities=None,
                 alpha_pos=1, alpha_neg=None, _lambda=1,
                 beta=1, delta=None, w_fun='tk',
                 **kwargs):
        self.outcomes = np.asarray(outcomes.copy())
        self.probabilities = np.asarray(probabilities.copy())
        self.alpha_pos = alpha_pos
        if alpha_neg is None:
            self.alpha_neg = alpha_pos
        else:
            self.alpha_neg = alpha_neg
        self._lambda = _lambda
        self.beta = beta
        if w_fun.lower() in ['tversky_kahneman', 'tversky', 'kahneman', 'tk']:
            self.w_fun = 'tversky_kahneman'
        elif w_fun.lower() in ['prelec', 'p']:
            self.w_fun = 'prelec'
        elif w_fun.lower() in ['gonzalez_wu', 'gonzalez', 'wu', 'gw']:
            self.w_fun = 'gonzalez_wu'
        else:
            raise ValueError("Invalid w_fun type.")
        if self.w_fun in ['prelec', 'gonzalez_wu'] and delta is None:
            self.delta = 1
        else:
            self.delta = delta
        self.shape = self.outcomes.shape
        if self.shape != self.probabilities.shape:
            raise ValueError("outcomes and probabilities do not have the same shape.")
        self.utilities = []
        self.weights = []
        self.expected_utility = []

    def utility(self, x=None):
        return np.where(x >= 0,
                        np.power(x, self.alpha_pos),
                        -self._lambda * np.power(-x, self.alpha_neg))
    
    def weighting_tk(self, x=None):
        numerator = np.power(x, self.beta)
        denominator = np.power((numerator + np.power(1 - x, self.beta)), 1/self.beta)
        return numerator / denominator
    
    def weighting_p(self, x=None):
        return np.exp(-self.delta * np.power(-np.log(x), self.beta))
    
    def weighting_gw(self, x=None):
        numerator = self.delta * np.power(x, self.beta)
        denominator = numerator + np.power(1 - x, self.beta)
        return numerator / denominator
    
    def compute(self):
        # Determine the utilities of the potential outcomes, for each choice option and each trial.
        self.utilities = np.array([
            [
                self.utility(self.outcomes[i, j])
                for j in range(self.shape[1])
            ]
            for i in range(self.shape[0])
        ], dtype=object)
        # Select the appropriate weighting function.
        if self.w_fun == 'tversky_kahneman':
            weighting_fun = self.weighting_tk
        elif self.w_fun == 'prelec':
            weighting_fun = self.weighting_p
        elif self.w_fun == 'gonzalez_wu':
            weighting_fun = self.weighting_gw
        # Determine the weights of the potential outcomes, for each choice option and each trial.
        self.weights = np.array([
            [
                weighting_fun(self.probabilities[i, j])
                for j in range(self.shape[1])
            ]
            for i in range(self.shape[0])
        ], dtype=object)
        # Determine the expected utility of each choice option for each trial.
        self.expected_utility = np.array([
            [
                np.sum(self.weights[i, j] * self.utilities[i, j])
                for j in range(self.shape[1])
            ]
            for i in range(self.shape[0])            
        ])
        return self.expected_utility

    def __call__(self):
        return self.compute()

    def __repr__(self):
        return f"{self.__class__.__name__}(outcomes={self.outcomes}, probabilities={self.probabilities}, alpha_pos={self.alpha_pos}, alpha_neg={self.alpha_neg}, _lambda={self._lambda}, beta={self.beta}, delta={self.delta},w_fun={self.w_fun})"

    def __str__(self):
        return f"{self.__class__.__name__}(outcomes={self.outcomes}, probabilities={self.probabilities}, alpha_pos={self.alpha_pos}, alpha_neg={self.alpha_neg}, _lambda={self._lambda}, beta={self.beta}, delta={self.delta},w_fun={self.w_fun})"

    def config(self):
        config = {
            "outcomes": self.outcomes,
            "probabilities": self.probabilities,
            "alpha_pos": self.alpha_pos,
            "alpha_neg": self.alpha_neg,
            "_lambda": self._lambda,
            "beta": self.beta,
            "delta": self.delta,
            "w_fun": self.w_fun,
            "type": "Activation",
            "name": self.__class__.__name__,
        }
        return config
