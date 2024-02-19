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

    where w is a weighting function of the probability of a potential outcome p, and u is the utility function of a potential outcome x.
    These functions are defined as follows (equations 6 and 5 respectively in Tversky & Kahneman, 1992, pp. 309):

        w(p) = p^beta / (p^beta + (1 - p)^beta)^(1/beta),
        u(x) = ifelse(x >= 0, x^alpha_pos, -lambda * (-x)^alpha_neg),

    where beta is the discriminability parameter of the weighting function;
    alpha_pos and alpha_neg are the risk attitude parameters in the gain and loss domains respectively,
    and lambda is the loss aversion parameter.

    Several other definitions of the weighting function have been proposed in the literature, most notably in Prelec (1998) and Gonzalez & Wu (1999).
    Prelec (equation 3.2, 1999, pp. 503) proposed the following definition:

        w(p) = exp(-delta * (-log(p))^beta),
    
    where delta and beta are the attractiveness and discriminability parameters of the weighting function.
    Gonzalez & Wu (equation 3, 1999, pp. 139) proposed the following definition:

        w(p) = (delta * p^beta) / ((delta * p^beta) + (1-p)^beta).

    Parameters
    ----------
    outcomes : array_like
        The values of potential outcomes for each choice option. Should be....
    probabilities : array_like
        The probabilities of potential outcomes for each choice option. Should be...
    alpha_pos : float
        The risk attitude parameter for non-negative outcomes, which determines the curvature of the utility function in the gain domain.
    alpha_neg : float
        The risk attitude parameter for negative outcomes, which determines the curvature of the utility function in the loss domain.
    lambda : float
        The loss aversion parameter, which scales the utility of negative outcomes relative to non-negative outcomes.
    beta : float
        The discriminability parameter, which determines the curvature of the weighting function.
    delta : float
        The attractiveness parameter, which determines the elevation of the weighting function.
    w_fun : str
        The definition of the weighting function. Should be...  

    Attributes
    ----------
    outcomes : array_like
        The values of potential outcomes for each choice option. Should be....
    probabilities : array_like
        The probabilities of potential outcomes for each choice option. Should be...
    alpha_pos : float
        The risk attitude parameter for non-negative outcomes, which determines the curvature of the utility function in the gain domain.
    alpha_neg : float
        The risk attitude parameter for negative outcomes, which determines the curvature of the utility function in the loss domain.
    loss_sc : float
        The loss aversion parameter (lambda), which scales the utility of negative outcomes relative to non-negative outcomes.
    beta : float
        The discriminability parameter, which determines the curvature of the weighting function.
    delta : float
        The attractiveness parameter, which determines the elevation of the weighting function.
    w_fun : str
        The definition of the weighting function. Should be...
    utilities : array_like
        The utility for each potential outcome.
    weights : array_like
        The weight for each potential outcome.
    expected_utility : array_like
        The expected utility for each choice option.

    Examples
    --------
    >>> outcomes = np.array()
    >>> probabilities = np.array()
    >>> prospect = ProspectUtility(
            outcomes, probabilities,
            alpha_pos = 0.85, loss_sc = 1.5,
            beta = 0.6, delta = 1
        )
    >>> prospect.compute()
    array([[0.03333333, 0.6       , 0.        ],
           [0.2       , 0.13333333, 0.        ]])

    References
    ----------
    Gonzalez, R., & Wu, G. (1999). On the shape of the probability weighting function. Cognitive psychology, 38(1), 129-166.

    Prelec, D. (1998). The probability weighting function. Econometrica, 497-527.

    Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and uncertainty, 5, 297-323.
    """

    def __init__(self, outcomes=None, probabilities=None, alpha_pos=1, alpha_neg=None, loss_sc=1, beta=1, delta=None, w_fun = 'tk', **kwargs):
        self.outcomes = outcomes
        self.probabilities = probabilities
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.loss_sc = loss_sc
        self.beta = beta
        self.delta = delta
        self.w_fun = w_fun
        self.utilities = []
        self.weights = []
        self.expected_utility = []

    def compute(self):
        """
        Compute each option's expected utility according to prospect theory, given the potential outcomes and their probabilities.

        Returns
        -------
        array_like
            The expected utilities of the choice options.
        """
        w_fun_list = ['tversky_kahneman', 'tversky', 'kahneman', 'tk', 'prelec', 'p', 'gonzalez_wu', 'gonzalez', 'wu', 'gw']
        if self.w_fun not in w_fun_list:
            raise ValueError("Invalid w_fun type. Expected one of: %s" % w_fun_list)
        
        
        if self.w_fun in w_fun_list[4:5]:
            self.weights = 'blabla'
        elif self.w_fun in w_fun_list[6:len(w_fun_list)]:
            self.weights = 'blabla'
        else:
            self.weights = 'blabla'

        return self.expected_utility

    def __call__(self):
        return self.compute()

    def __repr__(self):
        return f"{self.__class__.__name__}(outcomes={self.outcomes}, probabilities={self.probabilities}, alpha_pos={self.alpha_pos}, alpha_neg={self.alpha_neg}, loss_sc={self.loss_sc}, beta={self.beta}, delta={self.delta},w_fun={self.w_fun})"

    def __str__(self):
        return f"{self.__class__.__name__}(outcomes={self.outcomes}, probabilities={self.probabilities}, alpha_pos={self.alpha_pos}, alpha_neg={self.alpha_neg}, loss_sc={self.loss_sc}, beta={self.beta}, delta={self.delta},w_fun={self.w_fun})"

    def config(self):
        """
        Print the configuration of the prospect utility function.

        Returns
        -------
        dict
            The configuration of the prospect utility function.
        """
        return {
            "outcomes": self.outcomes,
            "probabilities": self.probabilities,
            "alpha_pos": self.alpha_pos,
            "alpha_neg": self.alpha_neg,
            "loss_sc": self.loss_sc,
            "beta": self.beta,
            "delta": self.delta,
            "w_fun": self.w_fun,
            "type": "Activation",
            "name": self.__class__.__name__,
        }
