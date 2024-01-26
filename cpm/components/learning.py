import numpy as np
from prettyformatter import pprint as pp

class DeltaRule:
    """
    DeltaRule class computes the prediction error for a given input and target value.
    It is based on the Gluck and Bower's (1988) delta rule, an extension to Rescorla
    and Wagner (1972), which was identical to that of Widrow and Hoff (1960).

    Attributes:
        alpha (float): The learning rate.
        weights (ndarray): The weight matrix.
        teacher (ndarray): The target values.
        input (ndarray): The input values.
        bounds (list): The bounds of the weight values.

    Methods:
        compute(): Updates the weight matrix based on the Delta rule.
        reset(): Resets the weight matrix to zeros.
        __repr__(): Returns a string representation of the DeltaRule object.
        config(): Returns a dictionary with the configuration of the DeltaRule object.
    """

    def __init__(self, alpha=None, weights=None, feedback=None, input=None, **kwargs):
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
        for i in range(self.shape[0]):
            activations = np.sum(self.weights[i])
            for j in range(self.shape[1]):
                # pp(self.alpha * (self.teacher[i] - activations) * self.input[j])
                self.weights[i, j] = self.alpha * (self.teacher[i] - activations) * self.input[j]
        return self.weights

    def reset(self):
        self.weights = np.zeros(self.shape)

    def __repr__(self):
        return f"DeltaRule(alpha={self.alpha},\n weights={self.weights},\n teacher={self.teacher})"

    def config(self):
        config = {
            'alpha': self.alpha,
            'weights': self.weights,
            'teacher': self.teacher,
            'name': self.__class__.__name__,
            'type': 'learning',
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
                output[i] += self.alpha * (self.reward + self.gamma * np.max(self.weights[i, :]) - self.weights[i, j]) * active[j]
        return output

class AttentionGateLearning:
    
    def __init__(self, theta, weights, input, teacher, attention, *args, **kwargs):
        self.theta = theta
        self.weights = weights
        self.input = input
        self.teacher = teacher
        self.attention = attention