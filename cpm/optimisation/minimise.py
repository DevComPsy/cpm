from scipy.stats import norm
import numpy as np

__all__ = ['LogLikelihood', 'BIC', 'CrossEntropy']

# Define your custom objective function
def LogLikelihood(predicted = None, observed = None, negative = True):
    """
    Compute the log likelihood of the predicted values given the observed values.

    Parameters
    ----------
    predicted : array-like
        The predicted values.
    observed : array-like
        The observed values.
    negative : bool, optional
        Flag indicating whether to return the negative log likelihood.

    Returns
    -------
    float
        The log likelihood or negative log likelihood.
    """
    
    # Compute the negative log likelihood
    LL = np.sum(norm.logpdf(predicted, observed, 0.5))
    if negative:
        LL = -1*LL
    return LL

def BIC(predicted, observed, n, k):
    """
    Calculate the Bayesian Information Criterion (BIC).

    Parameters
    ----------
    predicted : array-like
        The predicted values.
    observed : array-like
        The observed values.
    n : int
        The number of data points.
    k : int
        The number of parameters.

    Returns
    -------
    float
        The BIC value.
    """
    bic = -2 * LogLikelihood(predicted, observed, False) + k * np.log(n)
    return bic

def CrossEntropy(predicted, observed):
    """
    Calculate the cross entropy.

    Parameters
    ----------
    predicted : numpy.ndarray
        The predicted values.
    observed : numpy.ndarray
        The observed values.

    Returns
    -------
    float
        The cross entropy value.
    """
    ce = np.sum(-observed * np.log(predicted) - (1 - observed) * np.log(1 - predicted))
    return ce

# # Example data
# x = np.array([1, 2, 3, 4, 5])
# y_obs = np.array([2, 4, 6, 8, 10])

# # Initial guess for the parameters
# initial_params = np.array([1, 1])

# # Compute the predicted values
# y_pred = initial_params[0] * x + initial_params[1]

# log_likelihood(y_pred, y_obs)
# bic(log_likelihood(y_pred, y_obs), len(x), len(initial_params))

# # Extract the optimized parameters
# optimized_params = result.x

# print("Optimized parameters:", optimized_params)
