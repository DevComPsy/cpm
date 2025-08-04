import numpy as np
import pytest
from cpm.optimisation.minimise import Bayesian, LogLikelihood

class TestBayesian:
    def test_bic_basic(self):
        likelihood = -100.0
        n = 150
        k = 3
        expected = -2 * likelihood + k * np.log(n)
        result = Bayesian.BIC(likelihood, n, k)
        assert np.isclose(result, expected)

    def test_aic_basic(self):
        likelihood = -100.0
        n = 150  # n is not used in AIC, but included for interface
        k = 3
        expected = -2 * likelihood + 2 * k
        result = Bayesian.AIC(likelihood, n, k)
        assert np.isclose(result, expected)

    def test_bic_zero_likelihood(self):
        likelihood = 0.0
        n = 10
        k = 1
        expected = -2 * likelihood + k * np.log(n)
        result = Bayesian.BIC(likelihood, n, k)
        assert np.isclose(result, expected)

    def test_aic_zero_likelihood(self):
        likelihood = 0.0
        n = 10
        k = 1
        expected = -2 * likelihood + 2 * k
        result = Bayesian.AIC(likelihood, n, k)
        assert np.isclose(result, expected)

    def test_bic_invalid_n(self):
        likelihood = -10.0
        n = 0
        k = 2
        with pytest.raises(ValueError):
            Bayesian.BIC(likelihood, n, k)

    def test_bic_invalid_k(self):
        likelihood = -10.0
        n = 10
        k = -1
        with pytest.raises(ValueError):
            Bayesian.BIC(likelihood, n, k)

    def test_aic_invalid_k(self):
        likelihood = -10.0
        n = 10
        k = -1
        with pytest.raises(ValueError):
            Bayesian.AIC(likelihood, n, k)

class TestLogLikelihood:
    def test_categorical(self):
        expected = np.float64(-1.7350011354094463)
        observed = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        predicted = np.array([[0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
        result = LogLikelihood.categorical(predicted, observed, negative=False)
        assert isinstance(result, float)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_categorical_perfect(self):
        observed = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        predicted = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        result = LogLikelihood.categorical(predicted, observed, negative=False)
        assert isinstance(result, float)
        assert np.isclose(result, 0.0), "Perfect prediction should yield zero log likelihood"

    def test_categorical_zero_probabilities(self):
        expected = np.float64(-921.0340371976183)
        observed = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        predicted = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        result = LogLikelihood.categorical(predicted, observed, negative=False)
        assert isinstance(result, float)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_categorical_nan_handling(self):
        observed = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        predicted = np.array([[np.nan, 1.0], [0, np.nan], [0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            LogLikelihood.categorical(predicted, observed, negative=False)

    def test_bernoulli(self):
        expected = np.float64(-1.7350011354094463)
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([0.7, 0.3, 0.6, 0.4])
        result = LogLikelihood.bernoulli(predicted, observed, negative=False)
        assert isinstance(result, float)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_bernoulli_perfect(self):
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([1.0, 0.0, 1.0, 0.0])
        try:
            result = LogLikelihood.bernoulli(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"bernoulli raised {e}")
        assert isinstance(result, float)
        assert np.isclose(result, 0.0), "Perfect prediction should yield zero log likelihood"

    def test_bernoulli_zero_probabilities(self):
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([0.0, 1.0, 0.0, 1.0])
        # Should not raise and should return a float
        try:
            result = LogLikelihood.bernoulli(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"bernoulli raised {e}")
        assert isinstance(result, float)

    def test_bernoulli_nan_handling(self):
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([np.nan, 0.3, 0.6, 0.4])
        with pytest.raises(ValueError):
            LogLikelihood.bernoulli(predicted, observed, negative=False)

    def test_continuous(self):
        observed = np.array([1.0, 0.0, 1.0, 0.0])
        predicted = np.array([0.7, 0.3, 0.6, 0.4])
        result = LogLikelihood.continuous(predicted, observed, negative=False)
        assert isinstance(result, float)

    def test_continuous_perfect(self):
        expected = np.float64(-3.6757541328186907)
        observed = np.array([1.0, 0.0, 1.0, 0.0])
        predicted = np.array([1.0, 0.0, 1.0, 0.0])
        try:
            result = LogLikelihood.continuous(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"continuous raised {e}")
        assert isinstance(result, float)
        assert np.isclose(result, expected) 

    def test_continuous_zero_probabilities(self):
        expected = np.float64(-5.675754132818691)
        observed = np.array([1.0, 0.0, 1.0, 0.0])
        predicted = np.array([0.0, 1.0, 0.0, 1.0])
        # Should not raise and should return a float
        try:
            result = LogLikelihood.continuous(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"continuous raised {e}")
        assert isinstance(result, float)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_continuous_nan_handling(self):
        observed = np.array([1.0, 0.0, 1.0, 0.0])
        predicted = np.array([np.nan, 0.3, 0.6, 0.4])
        with pytest.raises(ValueError):
            LogLikelihood.continuous(predicted, observed, negative=False)

    def test_multinomial(self):
        expected = np.float64(-4.596597454123483)
        predicted = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
        observed = np.array([[2, 5, 3], [1, 7, 2]])
        # Should not raise and should return a float
        try:
            result = LogLikelihood.multinomial(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"multinomial raised {e}")
        assert isinstance(result, float)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_multinomial_perfect(self):
        expected = np.float64(-4.596597454123483)
        predicted = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
        observed = np.array([[2, 5, 3], [1, 7, 2]])
        try:
            result = LogLikelihood.multinomial(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"multinomial raised {e}")
        assert isinstance(result, float)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_multinomial_zero_probabilities(self):
        observed = np.array([[2, 5, 3], [1, 7, 2]])
        predicted = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        # Should not raise and should return a float
        try:
            result = LogLikelihood.multinomial(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"multinomial raised {e}")
        assert isinstance(result, float)

    def test_multinomial_nan_handling(self):
        observed = np.array([[2, 5, 3], [1, 7, 2]])
        predicted = np.array([[np.nan, 1.0, 0.0], [0.0, np.nan, 0.0]])
        with pytest.raises(ValueError):
            LogLikelihood.multinomial(predicted, observed, negative=False)

    def test_product(self):
        expected = np.float64(-0.8675005677047232)
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([0.7, 0.3, 0.6, 0.4])
        result = LogLikelihood.product(predicted, observed, negative=False)
        assert isinstance(result, float)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_product_perfect(self):
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([1.0, 0.0, 1.0, 0.0])
        try:
            result = LogLikelihood.product(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"product raised {e}")
        assert isinstance(result, float)
        assert np.isclose(result, 0.0), "Perfect prediction should yield zero log likelihood"

    def test_product_zero_probabilities(self):
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([0.0, 1.0, 0.0, 1.0])
        # Should not raise and should return a float
        try:
            result = LogLikelihood.product(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"product raised {e}")
        assert isinstance(result, float)

    def test_product_nan_handling(self):
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([np.nan, 0.3, 0.6, 0.4])
        with pytest.raises(ValueError):
            LogLikelihood.product(predicted, observed, negative=False)

if __name__ == "__main__":
    pytest.main()
