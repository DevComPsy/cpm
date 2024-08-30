from scipy.stats import norm, bernoulli
import numpy as np

__all__ = ["LogLikelihood", "Bayesian", "CrossEntropy"]


# Define your custom objective function
class LogLikelihood:

    def __init__(self) -> None:
        pass

    def categorical(predicted=None, observed=None, negative=True, **kwargs):
        """
        Compute the log likelihood of the predicted values given the observed values for categorical data.

            Categorical(y|p) = p_y

        Parameters
        ----------
        predicted : array-like
            The predicted values. It must have the same shape as `observed`. See Notes for more details.
        observed : array-like
            The observed values. It must have the same shape as `predicted`. See Notes for more details.
        negative : bool, optional
            Flag indicating whether to return the negative log likelihood.

        Returns
        -------
        float
            The log likelihood or negative log likelihood.

        Notes
        -----

        `predicted` and `observed` must have the same shape.
        `observed` is a binary variable, so it can only take the values 0 or 1.
        If there are two choice options, then observed would have a shape of (n, 2) and predicted would have a shape of (n, 2).
        On each row of `observed`, the array would have a 1 in the column corresponding to the observed value and a 0 in the other column.

        Examples
        --------
        >>> import numpy as np
        >>> observed = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        >>> predicted = np.array([[0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
        >>> LogLikelihood.categorical(predicted, observed)
        1.7350011354094463
        """
        values = np.array(predicted * observed).flatten()
        values = values[values != 0]
        # Compute the negative log likelihood
        LL = np.sum(np.log(values))
        if negative:
            LL = -1 * LL
        return LL

    def bernoulli(predicted=None, observed=None, negative=True, **kwargs):
        """
        Compute the log likelihood of the predicted values given the observed values for Bernoulli data.

            Bernoulli(y|p) = p if y = 1 and 1 - p if y = 0

        Parameters
        ----------
        predicted : array-like
            The predicted values. It must have the same shape as `observed`. See Notes for more details.
        observed : array-like
            The observed values. It must have the same shape as `predicted`. See Notes for more details.
        negative : bool, optional
            Flag indicating whether to return the negative log likelihood.

        Returns
        -------
        float
            The summed log likelihood or negative log likelihood.

        Notes
        -----

        `predicted` and `observed` must have the same shape.
        `observed` is a binary variable, so it can only take the values 0 or 1.
        `predicted` must be a value between 0 and 1.
        Values are clipped to avoid log(0) and log(1).
        If we encounter any non-finite values, we set any log likelihood to the value of np.log(1e-100).

        Examples
        --------
        >>> import numpy as np
        >>> observed = np.array([1, 0, 1, 0])
        >>> predicted = np.array([0.7, 0.3, 0.6, 0.4])
        >>> LogLikelihood.bernoulli(predicted, observed)
        1.7350011354094463

        """
        limit = np.log(1e-200)
        bound = np.finfo(np.float64).min
        probabilities = predicted.flatten()
        np.clip(probabilities, 1e-100, 1 - 1e-100, out=probabilities)

        LL = bernoulli.logpmf(k=observed.flatten(), p=probabilities)
        LL[LL < bound] = limit  # Set the lower bound to avoid overflow
        LL = np.sum(LL)
        if negative:
            LL = -1 * LL
        return LL

    def continuous(predicted, observed, negative=True, **kwargs):
        """
        Compute the log likelihood of the predicted values given the observed values for continuous data.

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
            The summed log likelihood or negative log likelihood.

        Examples
        --------
        >>> import numpy as np
        >>> observed = np.array([1, 0, 1, 0])
        >>> predicted = np.array([0.7, 0.3, 0.6, 0.4])
        >>> LogLikelihood.continuous(predicted, observed)
        1.7350011354094463
        """
        LL = np.sum(norm.logpdf(predicted, observed, 1))
        if negative:
            LL = -1 * LL
        return LL


class Distance:

    def __init__(self):
        pass

    def SSE(predicted, observed, **kwargs):
        """
        Compute the sum of squared errors (SSE).

        Parameters
        ----------
        predicted : array-like
            The predicted values.
        observed : array-like
            The observed values.

        Returns
        -------
        float
            The sum of squared errors.
        """
        sse = np.sum((predicted.flatten() - observed.flatten()) ** 2)
        return sse

    def MSE(predicted, observed, **kwargs):
        """
        Compute the Mean Squared Errors (EDE).

        Parameters
        ----------
        predicted : array-like
            The predicted values.
        observed : array-like
            The observed values.

        Returns
        -------
        float
            The Euclidean distance.
        """
        euclidean = np.sqrt(np.mean((predicted.flatten() - observed.flatten()) ** 2))
        return euclidean


class Bayesian:

    def __init__(self) -> None:
        pass

    def BIC(likelihood, n, k, **kwargs):
        """
        Calculate the Bayesian Information Criterion (BIC).

        Parameters
        ----------
        likelihood : float
            The log likelihood value.
        n : int
            The number of data points.
        k : int
            The number of parameters.

        Returns
        -------
        float
            The BIC value.
        """
        bic = -2 * likelihood + k * np.log(n)
        return bic

    def AIC(likelihood, n, k, **kwargs):
        """
        Calculate the Akaike Information Criterion (AIC).

        Parameters
        ----------
        likelihood : float
            The log likelihood value.
        n : int
            The number of data points.
        k : int
            The number of parameters.

        Returns
        -------
        float
            The AIC value.
        """
        aic = -2 * likelihood + 2 * k
        return aic


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
