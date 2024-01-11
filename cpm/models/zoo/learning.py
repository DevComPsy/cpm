"""
Model components
"""
import numpy as np
from . import utils

__all__ = ['DeltaRule', 'HebbRule', 'QLearningRule']

class DeltaRule:
    """
    DeltaRule class represents a learning algorithm based on the Delta rule.

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

    def __init__(self, alpha=None, weights=None, teacher=None, input=None, **kwargs):
        self.alpha = alpha
        self.weights = weights
        self.teacher = teacher
        self.input = input
        self.bounds = [0, 1]

    def compute(self):
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] += self.alpha * (self.teacher[i] - np.sum(self.weights[i])) * self.input[j]
        return self.weights

    def reset(self):
        self.weights = np.zeros(self.weights.shape)

    def __repr__(self):
        return f"DeltaRule(alpha={self.alpha},\n weights={self.weights},\n teacher={self.teacher})"

    def config(self):
        config = {
            'alpha': self.alpha,
            'weights': self.weights,
            'teacher': self.teacher,
            'name': self.__class__.__name__,
            'bounds': self.bounds,
            'type': 'learning'
        }
        return config

# arguments = {
#     'alpha' : 0.1,
#     'weights' : np.array([[0.5, 0], [0, 0.5]]),
#     'input' : np.array([1, 1]),
#     'teacher' : np.array([1, 0]),
#     'attention' : np.array([1, 0]),
#     'misc' : np.array([1, 0])
#     }

# input = arguments

# arguments['weights']

# x = DeltaRule(**input)
# x.weights
# x.teacher
# x.input
# x.compute()
# x.compute()
# x.reset()
# x.weights
# # pf.pprint(x.config())

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