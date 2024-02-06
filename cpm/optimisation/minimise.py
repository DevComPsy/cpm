from scipy.stats import norm
import numpy as np

__all__ = ["LogLikelihood", "BIC", "CrossEntropy"]


# Define your custom objective function
def LogLikelihood(predicted=None, observed=None, negative=True, **kwargs):
    # FIXME: log likelihood sigma is not correct
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
        LL = -1 * LL
    return LL


def BIC(predicted, observed, n, k, **kwargs):
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


def CrossEntropy(predicted, observed, **kwargs):
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
    ce = np.sum(-observed * np.log(predicted) + (1 - observed) * np.log(1 - predicted))
    return ce
