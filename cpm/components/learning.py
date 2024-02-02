import numpy as np


class DeltaRule:
    """
    DeltaRule class computes the prediction error for a given input and target value.
    It is based on the Gluck and Bower's (1988) delta rule, an extension to Rescorla
    and Wagner (1972), which was identical to that of Widrow and Hoff (1960).

    The delta-rule is a summed error term, which means that the error is defined as
    the difference between the target value and the summed activation of all values
    for a given output unit. For separable error term, see the Bush and Mosteller (1951) rule.

    Attributes
    ----------
    alpha : float
        The learning rate.
    input : ndarray or array_like
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    weights : ndarray
        The weights value. A 2D array of weights, where each row represents an outcome and each column represents a single stimulus.
    teacher : ndarray
        The target values or feedback, sometimes referred to as teaching signals. These are the values that the algorithm should learn to predict.
    shape : tuple
        The shape of the weight matrix.

    Parameters
    ----------
    alpha : float
        The learning rate.
    weights : array-like
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    feedback : array-like
        The target values or feedback, sometimes referred to as teaching signals. These are the values that the algorithm should learn to predict.
    input : array-like
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    **kwargs : dict, optional
        Additional keyword arguments.

    See Also
    --------
    [cpm.components.learning.SeparableRule][cpm.components.learning.SeparableRule] : A class representing a learning rule based on the separable error-term of Bush and Mosteller (1951).

    Examples
    --------
    >>> import numpy as np
    >>> from cpm.components.learning import DeltaRule
    >>> weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> teacher = np.array([1, 0])
    >>> input = np.array([1, 1, 0])
    >>> delta_rule = DeltaRule(alpha=0.1, weights=weights, feedback=teacher, input=input)
    >>> delta_rule.compute()
    array([[ 0.04,  0.04,  0.  ],
           [-0.15, -0.15, -0.  ]])

    This implementation generalises to n-dimensional weight matrices, which means
    that it can be applied to both single- and multi-outcome learning paradigms.

    >>> weights = np.array([0.1, 0.6, 0., 0.3])
    >>> teacher = np.array([1])
    >>> input = np.array([1, 1, 0, 0])
    >>> delta_rule = DeltaRule(alpha=0.1, weights=weights, feedback=teacher, input=input)
    >>> delta_rule.compute()
    array([[0.03, 0.03, 0.  , 0.  ]])
    """

    def __init__(self, alpha=None, weights=None, feedback=None, input=None, **kwargs):
        """ """
        self.alpha = alpha
        self.weights = [[]]
        if weights is not None:
            self.weights = np.asarray(weights.copy())
        self.teacher = feedback
        self.input = np.asarray(input)
        self.shape = self.weights.shape
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])
            self.weights = np.array([self.weights])

    def compute(self):
        """
        Compute the weights using the CPM learning rule.

        Returns
        -------
        weights: numpy.ndarray
            The updated weights matrix.
        """

        for i in range(self.shape[0]):
            activations = np.sum(self.weights[i] * self.input)
            for j in range(self.shape[1]):
                self.weights[i, j] = (
                    self.alpha * (self.teacher[i] - activations) * self.input[j]
                )
        return self.weights

    def reset(self):
        """
        Reset the weights to zero.
        """
        self.weights = np.zeros(self.shape)

    def __repr__(self):
        return f"DeltaRule(alpha={self.alpha},\n weights={self.weights},\n teacher={self.teacher})"

    def config(self):
        """
        Get the configuration of the learning component.

        Returns
        -------
        config: dict
            A dictionary containing the configuration parameters of the learning component.

            - alpha (float): The learning rate.
            - weights (list): The weights used for learning.
            - teacher (str): The name of the teacher.
            - name (str): The name of the learning component class.
            - type (str): The type of the learning component.
        """
        config = {
            "alpha": self.alpha,
            "weights": self.weights,
            "teacher": self.teacher,
            "name": self.__class__.__name__,
            "type": "learning",
        }
        return config


class SeparableRule:
    """
    A class representing a learning rule based on the separable error-term of
    Bush and Mosteller (1951).

    Parameters
    -----------
    alpha : float, optional
        The learning rate.
    weights : array-like, optional
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    feedback : array-like, optional
        The target values or feedback, sometimes referred to as teaching signals. These are the values that the algorithm should learn to predict.
    input : array-like, optional
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    **kwargs : dict, optional
        Additional keyword arguments.

    Attributes
    -----------
    alpha : float
        The learning rate.
    input : ndarray
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    weights : ndarray
        The weights value. A 2D array of weights, where each row represents an outcome and each column represents a single stimulus.
    teacher : ndarray
        The target values or feedback, sometimes referred to as teaching signals. These are the values that the algorithm should learn to predict.
    shape : tuple
        The shape of the weights array.

    See Also
    --------
    [cpm.components.learning.DeltaRule][cpm.components.learning.DeltaRule] : An extension of the Rescorla and Wagner (1972) learning rule by Gluck and Bower (1988) to allow multi-outcome learning.
    """

    def __init__(self, alpha=None, weights=None, feedback=None, input=None, **kwargs):
        """
        Initializes the SeparableRule object.

        Parameters
        ----------
        alpha : float
            The learning rate.
        weights : ndarray
            The weight matrix.
        feedback : ndarray
            The target values.
        input : ndarray
            The input values.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.alpha = alpha
        self.weights = [[]]
        if weights is not None:
            self.weights = weights.copy()
        self.teacher = feedback
        self.input = np.asarray(input)
        self.shape = self.weights.shape
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])
            self.weights = np.array([self.weights])

    def compute(self):
        """
        Computes the updated weights using the learning rule.

        Returns:
        --------
        ndarray
            The updated weights.
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.weights[i, j] = (
                    self.alpha * (self.teacher[i] - self.weights[i, j]) * self.input[j]
                )
        return self.weights

    def reset(self):
        """
        Resets the weights to zero.
        """
        self.weights = np.zeros(self.shape)

    def __repr__(self):
        """
        Returns a string representation of the object.
        """
        return f"SeparableRule(alpha={self.alpha},\n weights={self.weights},\n teacher={self.teacher})"

    def config(self):
        """
        Returns a dictionary containing the configuration of the object.

        Returns:
        --------
        dict
            The configuration of the object.
        """
        config = {
            "alpha": self.alpha,
            "weights": self.weights,
            "teacher": self.teacher,
            "name": self.__class__.__name__,
            "type": "learning",
        }
        return config


# NOTE: NOT TESTED
class HebbRule:
    def __init__(self, alphas, weights, *args, **kwargs):
        self.weights = weights
        self.alpha = alpha

    def call(self):
        active = self.weights
        active[active > 0] = 1

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] += self.alpha * active[i] * active[j]
        return self.weights


class QLearningRule:
    """
    Q-learning rule for reinforcement learning.

    Parameters
    ----------
    alpha : float
        The learning rate. Default is 0.5.
    gamma : float
        The discount factor. Default is 0.1.
    values : ndarray
        The values matrix.  It is a 1D array of Q-values active for the current state, where each element corresponds to an action.
    reward : float
        The reward received on the current state.
    maximum : float
        The maximum estimated reward for the next state.

    Attributes
    ----------
    alpha : float
        The learning rate.
    gamma : float
        The discount factor.
    values : ndarray
        The values matrix. It is a 1D array of Q-values, where each element corresponds to an action.
    reward : float
        The reward received on the current state.
    maximum : float
        The maximum estimated reward for the next state.
    """

    def __init__(
        self,
        alpha=0.5,
        gamma=0.1,
        values=None,
        reward=None,
        maximum=None,
        *args,
        **kwargs,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.values = values.copy()
        self.reward = reward
        self.maximum = maximum

    def compute(self):
        """
        Compute the change in values based on the given values, reward, and parameters, and return the updated values.

        Returns
        -------
        output: numpy.ndarray:
            The computed output values.
        """

        active = self.values.copy()
        active[active > 0] = 1
        output = np.zeros(self.values.shape[0])

        for i in range(self.values.shape[0]):
            output[i] += (1 - self.alpha) * self.values[i] + (
                self.alpha * (self.reward + self.gamma * self.maximum)
            ) * active[i]

        return output

    def __repr__(self):
        return f"QLearningRule(alpha={self.alpha},\n gamma={self.gamma},\n values={self.values},\n reward={self.reward},\n maximum={self.maximum})"

    def config(self):
        """
        Get the configuration of the q-learning component.

        Returns
        -------
        config: dict
            A dictionary containing the configuration parameters of the learning component.

            - alpha (float): The learning rate.
            - gamma (float): The discount factor.
            - values (list): The values used for learning.
            - reward (str): The name of the reward.
            - maximum (str): The name of the maximum reward.
            - name (str): The name of the learning component class.
            - type (str): The type of the learning component.
        """
        config = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "values": self.values,
            "reward": self.reward,
            "maximum": self.maximum,
            "name": self.__class__.__name__,
            "type": "learning",
        }
        return config


# input = np.array([1, 0.5, 0.99])

# component = QLearningRule(alpha=0.1, gamma=0.8, values=input, reward=1, maximum=10)
# component.values
# component.compute()

# class AttentionGateLearning:
#     def __init__(self, theta, weights, input, teacher, attention, *args, **kwargs):
#         self.theta = theta
#         self.weights = weights
#         self.input = input
#         self.teacher = teacher
#         self.attention = attention
