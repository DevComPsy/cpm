"""
Attentional processes for learning
"""

import numpy as np

class CompetitiveGating:

    def __init__(self, brutality = None, input = None, weights = None, attention = None, **kwargs):
        self.input = input
        self.attention = attention.copy()
        self.weights = weights.copy()
        self.brutality = brutality
        self.gain = None

    def compute(self):
        """ Competitive gating function
        """
        self.gain = self.input * self.attention
        self.gain = self.gain / np.sum(self.gain ** self.brutality) ** (1/self.brutality)
        for i in range(self.weights.shape[0]):
            self.weights[i] = self.weights[i] * self.gain[i]
        return self.gain

# input = {
#     'brutality' : 2,
#     'weights' : np.array([[0.5, 0.1], [0.9, 0.5]]),
#     'input' : np.array([1, 1]),
#     'attention' : np.array([1, 2]),
#     }

# x = CompetitiveGating(**input)
# x.compute()
# x.weights