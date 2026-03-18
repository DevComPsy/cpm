import numpy as np

__all = ["PenalisedLikelihoods"]

class PenalisedLikelihoods:

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
        if k < 0:
            raise ValueError("Number of parameters (k) must be non-negative.")
        aic = -2 * likelihood + 2 * k
        return aic