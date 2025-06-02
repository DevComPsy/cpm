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
        `observed` is a vector of integers starting from 0 (first possible response), where each integer corresponds to the observed value.
        If there are two choice options, then observed would have a shape of (n, 2) and predicted would have a shape of (n, 2).
        On each row of `observed`, the array would have a 1 in the column corresponding to the observed value and a 0 in the other column.

        Examples
        --------
        >>> import numpy as np
        >>> observed = np.array([0, 1, 0, 1])
        >>> predicted = np.array([[0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
        >>> LogLikelihood.categorical(predicted, observed)
        1.7350011354094463
        """
        observed_format = np.apply_along_axis(
            lambda x: np.eye(observed.max() + 1)[x], 0, observed
        )
        observed_format = np.concatenate(observed_format, axis=0).reshape(-1, 2)
        values = np.array(predicted * observed_format).flatten()
        values = values[values != 0]
        values = values.sum(axis=1)
        np.clip(values, 1e-100, 1 - 1e-100, out=values)
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

    def multinomial(predicted, observed, negative=True, clip=1e-10, **kwargs):
        """
        Compute the log likelihood of the predicted values given the observed values for multinomial data.

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
            The summed log likelihood or negative summed log likelihood.

        Examples
        --------
        >>> # Sample data
        >>> predicted = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
        >>> observed = np.array([[2, 5, 3], [1, 7, 2]])

        >>> # Calculate log likelihood
        >>> ll_float = LogLikelihood.multinomial(predicted, observed)
        >>> print("Log Likelihood (multinomial):", ll)
        Log Likelihood (multinomial): 4.596597454123483
        """
        if isinstance(observed, np.ndarray) and len(observed) == 1:
            observed = np.array(observed[0], dtype=float)
        else:
            observed = np.array(observed, dtype=float)
        predicted, observed = np.squeeze(predicted), np.squeeze(observed)
        if predicted.shape != observed.shape:
            raise ValueError("The predicted and observed values must have the same shape.")
        if not np.allclose(predicted.sum(axis=-1), 1):
            raise ValueError("The predicted values must sum to 1 within a tolerance.")
        predicted = np.clip(predicted, clip, np.inf)  # Avoid log(0)
        predicted = predicted / predicted.sum(axis=-1, keepdims=True)
        LL = np.sum(
            [
                multinomial.logpmf(observed[i], n=observed[i].sum(), p=predicted[i])
                for i in range(observed.shape[0])
            ]
        )
        if negative:
            LL = -1 * LL
        return LL

    def product(predicted, observed, negative=True, clip=1e-10, **kwargs):
        """
        Compute the log likelihood of the predicted values given the observed values for continuous data,
        according to the following equation:

            likelihood = sum(observed * log(predicted))

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
        >>> # Sample data
        >>> predicted = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
        >>> observed = np.array([[2, 5, 3], [1, 7, 2]])

        >>> # Calculate log likelihood
        >>> ll_float = LogLikelihood.product(predicted, observed)
        >>> print("Log Likelihood :", ll_float)
        Log Likelihood : 18.314715666079106
        """
        limit = np.log(1e-200)
        bound = np.finfo(np.float64).min
        predicted, observed = np.squeeze(predicted), np.squeeze(observed)
        predicted = np.clip(predicted, clip, np.inf)  # Avoid log(0)
        LL = observed.flatten() * np.log(predicted.flatten())
        ## swap NA with -Inf
        LL[np.isnan(LL)] = -np.inf
        LL[LL < bound] = limit  # Set the lower bound to avoid overflow
        LL = np.sum(LL)
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
