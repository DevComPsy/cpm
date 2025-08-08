from scipy.stats import norm, bernoulli, multinomial
import numpy as np

__all__ = ["LogLikelihood", "Bayesian", "CrossEntropy"]


def check_nan_and_bounds_in_input(predicted, observed):
    """
    Check if predicted or observed values contain NaN, Inf, nothing, and also check for shape mismatching.
    Accepts array-like inputs.
    If any of the above is found, raise an error with details.
    Returns squeezed arrays for shape compatibility.
    """
    if predicted is None or observed is None:
        raise ValueError("Predicted and observed values must not be None.")

    if not isinstance(predicted, (np.ndarray, list)) or not isinstance(observed, (np.ndarray, list)):
        raise TypeError("Predicted and observed values must be array-like (numpy.ndarray or list).")
    if len(predicted) == 0:
        raise ValueError("Model output must not be empty.")
    if len(observed) == 0:
        raise ValueError("Observed values must not be empty.")
    predicted = np.asarray(predicted)
    predicted = np.asarray(predicted)
    observed = np.asarray(observed)
    if predicted.size == 0:
        raise ValueError("Model output must not be empty.")
    if observed.size == 0:
        raise ValueError("Observed values must not be empty.")
    predicted_s = np.squeeze(predicted)
    observed_s = np.squeeze(observed)
    if np.any(np.isnan(predicted_s)):
        idx = np.where(np.isnan(predicted_s))
        raise ValueError(f"Predicted values contain NaN at indices {idx}.")
    if np.any(np.isnan(observed_s)):
        idx = np.where(np.isnan(observed_s))
        raise ValueError(f"Observed values contain NaN at indices {idx}.")
    if np.any(np.isinf(predicted_s)):
        idx = np.where(np.isinf(predicted_s))
        raise ValueError(f"Predicted values contain Inf at indices {idx}.")
    if np.any(np.isinf(observed_s)):
        idx = np.where(np.isinf(observed_s))
        raise ValueError(f"Observed values contain Inf at indices {idx}.")
    if predicted_s.shape != observed_s.shape:
        raise ValueError(f"Shape mismatch: predicted shape {predicted.shape} (squeezed {predicted_s.shape}) does not match observed shape {observed.shape} (squeezed {observed_s.shape}).")
    return predicted_s, observed_s

def check_nan_bounds_in_log(value, bound=-1e100):
    """
    Check if the value is NaN, Inf, or out of bounds, and then replace it with a bound value.

    Parameters
    ----------
    value : array-like
        The value(s) to check.
    bound : float, optional
        The value to exchange NaN, Inf, and out-of-bounds with, by default -1e100.
    """
    output = np.asarray(value, copy=True)
    output = np.nan_to_num(output, copy=False, nan=bound, posinf=bound, neginf=bound)
    return output

# Define your custom objective function
class LogLikelihood:

    def __init__(self) -> None:
        pass

    @staticmethod
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
        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)
        bound = -1e100
        values = np.array(predicted * observed).flatten()
        values = values[observed.flatten() != 0]
        ## bump up the probabilities to avoid log(0)
        np.clip(values, 1e-100, 1 - 1e-100, out=values)
        LL = np.log(values)
        LL = check_nan_bounds_in_log(LL, bound=bound)
        # Compute the negative log likelihood
        LL = np.sum(LL)
        if negative:
            LL = -1 * LL
        return LL

    @staticmethod
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
        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)
        bound = -1e100
        probabilities = predicted.flatten()
        ## bump up the probabilities to avoid log(0)
        np.clip(probabilities, 1e-100, 1 - 1e-100, out=probabilities)

        LL = bernoulli.logpmf(k=observed.flatten(), p=probabilities)
        LL = check_nan_bounds_in_log(LL, bound=bound)
        LL = np.sum(LL)
        if negative:
            LL = -1 * LL
        return LL

    @staticmethod
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

        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)
        bound = -1e100
        LL = norm.logpdf(predicted, observed, 1)
        LL = check_nan_bounds_in_log(LL, bound=bound)
        LL = np.sum(LL)
        if negative:
            LL = -1 * LL
        return LL

    @staticmethod
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
        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)
        bound = -1e100
        LL = multinomial.logpmf(observed, n=np.sum(observed, axis=1), p=predicted)
        LL = check_nan_bounds_in_log(LL, bound=bound)
        LL = np.sum(LL)
        if negative:
            LL = -1 * LL
        return LL

    @staticmethod
    def product(predicted, observed, negative=True, clip=1e-10, **kwargs):
        """
        Compute the log likelihood of the predicted values given the observed values for continuous data, according to the following equation:

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
        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)

        bound = -1e100
        np.clip(predicted, 1e-100, 1 - 1e-100, out=predicted)
        LL = observed.flatten() * np.log(predicted.flatten())
        LL = check_nan_bounds_in_log(LL, bound=bound)
        LL = np.sum(LL)
        if negative:
            LL = -1 * LL
        return LL


class Distance:

    def __init__(self):
        pass

    @staticmethod
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
        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)
        sse = np.sum((predicted.flatten() - observed.flatten()) ** 2)
        sse = np.float64(sse)  # Ensure the result is a float
        return sse

    @staticmethod
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
        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)
        euclidean = np.mean((predicted.flatten() - observed.flatten()) ** 2)
        return euclidean

    @staticmethod
    def RMSE(predicted, observed, **kwargs):
        """
        Compute the Root Mean Squared Errors (RMSE).

        Parameters
        ----------
        predicted : array-like
            The predicted values.
        observed : array-like
            The observed values.

        Returns
        -------
        float
            The Root Mean Squared Errors.
        """
        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)
        rmse = np.sqrt(np.mean((predicted.flatten() - observed.flatten()) ** 2))
        return rmse

class Discrete:

    def __init__(self) -> None:
        pass

    @staticmethod
    def ChiSquare(predicted, observed, **kwargs):
        """
        Compute the Chi-Square statistic.

        Parameters
        ----------
        predicted : array-like
            The predicted values.
        observed : array-like
            The observed values.

        Returns
        -------
        float
            The Chi-Square statistic.
        """
        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)
        predicted = np.array(predicted, dtype=float)
        observed = np.array(observed, dtype=float)
        chi_square = ((observed - (np.sum(observed) * predicted)) ** 2 / (np.sum(observed) * predicted))
        chi_square = np.sum(chi_square)
        return chi_square

    @staticmethod
    def G2(predicted, observed, **kwargs):
        """
        Compute the G2 statistic.

        Parameters
        ----------
        predicted : array-like
            The predicted values.
        observed : array-like
            The observed values.

        Returns
        -------
        float
            The G2 statistic.
        """
        predicted, observed = check_nan_and_bounds_in_input(predicted, observed)
        predicted = np.array(predicted, dtype=float)
        observed = np.array(observed, dtype=float)
        g2 = 2 * np.sum(observed * np.log(observed / (np.sum(observed) * predicted)))
        return g2

class Bayesian:

    def __init__(self) -> None:
        pass

    @staticmethod
    def BIC(likelihood: float, n: int, k: int, **kwargs) -> float:
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
        if n <= 0:
            raise ValueError("Number of data points (n) must be greater than 0.")
        if k < 0:
            raise ValueError("Number of parameters (k) must be non-negative.")
        if not isinstance(likelihood, (int, float)):
            raise TypeError("Likelihood must be a numeric value.")
        bic = -2 * likelihood + k * np.log(n)
        return bic

    @staticmethod
    def AIC(likelihood: float, n: int, k: int, **kwargs) -> float:
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
        if n <= 0:
            raise ValueError("Number of data points (n) must be greater than 0.")
        if k < 0:
            raise ValueError("Number of parameters (k) must be non-negative.")
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
    check_nan_and_bounds_in_input(predicted, observed)
    ce = np.sum(-observed * np.log(predicted) + (1 - observed) * np.log(1 - predicted))
    return ce
