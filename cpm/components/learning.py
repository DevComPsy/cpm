import numpy as np


class DeltaRule:
    """
    DeltaRule class computes the prediction error for a given input and target value.
    It is based on the Gluck and Bower's (1988) delta rule, an extension to Rescorla
    and Wagner (1972), which was identical to that of Widrow and Hoff (1960).

    The delta-rule is a summed error term, which means that the error is defined as
    the difference between the target value and the summed activation of all available
    output units. For separable error term, see the Bush and Mosteller (1951) rule.

    Attributes
    ----------
    alpha : float
        The learning rate.
    weights : ndarray
        The weight matrix.
    teacher : ndarray
        The target values.
    input : ndarray
        The input values.
    shape : tuple
        The shape of the weight matrix.
    """

    def __init__(self, alpha=None, weights=None, feedback=None, input=None, **kwargs):
        """
        Initializes the DeltaRule object.

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
            Compute the weights using the CPM learning rule.

            Returns:
                numpy.ndarray: The updated weights matrix.
            """

            for i in range(self.shape[0]):
                activations = np.sum(self.weights[i])
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

            Returns:
                dict: A dictionary containing the configuration parameters of the learning component.
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
        The learning rate. Default is None.
    weights : array-like, optional
        The initial weights. Default is None.
    feedback : array-like, optional
        The teacher feedback. Default is None.
    input : array-like, optional
        The input values. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments.

    Attributes
    -----------
    alpha : float
        The learning rate.
    weights : ndarray
        The weights.
    teacher : array-like
        The teacher feedback.
    input : ndarray
        The input values.
    shape : tuple
        The shape of the weights array.

    See Also
    --------
    SeparableRule : A class representing a learning rule based on the separable error-term of Bush and Mosteller (1951).
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
        return f"DeltaRule(alpha={self.alpha},\n weights={self.weights},\n teacher={self.teacher})"

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


# NOTE: NOT TESTED
class QLearningRule:
    def __init__(self, alpha, gamma, weights, reward, *args, **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.reward = reward

    def activate(self):
        active = self.weights
        active[active > 0] = 1
        output = np.zeros(self.weights.shape[0])

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                output[i] += (
                    self.alpha
                    * (
                        self.reward
                        + self.gamma * np.max(self.weights[i, :])
                        - self.weights[i, j]
                    )
                    * active[j]
                )
        return output


class AttentionGateLearning:
    def __init__(self, theta, weights, input, teacher, attention, *args, **kwargs):
        self.theta = theta
        self.weights = weights
        self.input = input
        self.teacher = teacher
        self.attention = attention