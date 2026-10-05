import numpy as np
import pytest
from cpm.optimisation.compare import PenalisedLikelihoods as PL

class TestPenalisedLikelihoods:
    def test_bic_basic(self):
        likelihood = -100.0
        n = 150
        k = 3
        expected = -2 * likelihood + k * np.log(n)
        result = PL.BIC(likelihood, n, k)
        assert np.isclose(result, expected)

    def test_aic_basic(self):
        likelihood = -100.0
        n = 150  # n is not used in AIC, but included for interface
        k = 3
        expected = -2 * likelihood + 2 * k
        result = PL.AIC(likelihood, n, k)
        assert np.isclose(result, expected)

    def test_bic_zero_likelihood(self):
        likelihood = 0.0
        n = 10
        k = 1
        expected = -2 * likelihood + k * np.log(n)
        result = PL.BIC(likelihood, n, k)
        assert np.isclose(result, expected)

    def test_aic_zero_likelihood(self):
        likelihood = 0.0
        n = 10
        k = 1
        expected = -2 * likelihood + 2 * k
        result = PL.AIC(likelihood, n, k)
        assert np.isclose(result, expected)

    def test_bic_invalid_n(self):
        likelihood = -10.0
        n = 0
        k = 2
        with pytest.raises(ValueError):
            PL.BIC(likelihood, n, k)

    def test_bic_invalid_k(self):
        likelihood = -10.0
        n = 10
        k = -1
        with pytest.raises(ValueError):
            PL.BIC(likelihood, n, k)

    def test_aic_invalid_k(self):
        likelihood = -10.0
        n = 10
        k = -1
        with pytest.raises(ValueError):
            PL.AIC(likelihood, n, k)

if __name__ == "__main__":
    pytest.main()