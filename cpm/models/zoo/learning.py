"""
Model components
"""
import numpy as np

__all__ = ['DeltaRule', 'HebbRule', 'QLearningRule']

class DeltaRule:
    def __init__(self, alpha=None, weights=None, teacher=None, ***args, **kwargs):
        self.alpha = alpha
        self.weights = weights
        self.teacher = teacher

    def activate(self):
        active = self.weights
        active[active > 0] = 1

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] += self.alpha * (self.teacher[i] - self.weights[i, j]) * active[i, j]
        return self.weights

class HebbRule:
    def __init__(self, weights, alpha, ***args, **kwargs):
        self.weights = weights
        self.alpha = alpha

    def activate(self):
        active = self.weights
        active[active > 0] = 1

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] += self.alpha * active[i] * active[j]
        return self.weights

class QLearningRule:
    def __init__(self, alpha, gamma, weights, reward, ***args, **kwargs):
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

# FIXME: needs to add references and match equation
def delta_rule(alpha = None, weights = None, teacher = None, ***args, **kwargs):
    """
    Delta rule
    """
    active = weights
    active[active > 0] = 1
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] += alpha * (teacher[i] - weights[i, j]) * active[i, j]
    return weights

# FIXME: this below needs to be double checked and added documentation
def hebb_rule(weights, alpha, ***args, **kwargs):
    """
    Hebb rule
    """
    active = weights
    active[active > 0] = 1
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] += alpha * active[i] * active[j]
    return weights

# FIXME: this below needs to be double checked and added documentation
def qlearning_rule( alpha, gamma, weights, reward, ***args, **kwargs):
    """
    Q-learning rule
    """
    active = weights
    active[active > 0] = 1
    output = np.zeros(weights.shape[0])
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            output[i] += alpha * (reward + gamma * np.max(weights[i, :]) - weights[i, j]) * active[j]
    return output

