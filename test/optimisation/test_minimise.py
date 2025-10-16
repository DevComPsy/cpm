import numpy as np
import pytest
from cpm.optimisation.minimise import Bayesian, LogLikelihood, Distance, Discrete

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
        expected = np.float64(-92.10340371976183)
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
        assert isinstance(result, float), "Expected float, got {type(result)}"
        assert np.isclose(result, 0.0), "Perfect prediction should yield zero log likelihood"

    def test_product_zero_probabilities(self):
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([0.0, 1.0, 0.0, 1.0])
        # Should not raise and should return a float
        try:
            result = LogLikelihood.product(predicted, observed, negative=False)
        except Exception as e:
            pytest.fail(f"product raised {e}")
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_product_nan_handling(self):
        observed = np.array([1, 0, 1, 0])
        predicted = np.array([np.nan, 0.3, 0.6, 0.4])
        with pytest.raises(ValueError):
            LogLikelihood.product(predicted, observed, negative=False)

class TestDistance:
    def test_sse_basic(self):
        predicted = np.array([1, 2, 3])
        observed = np.array([1, 2, 4])
        result = Distance.SSE(predicted, observed)
        assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"
        assert isinstance(result, np.float64), f"Expected np.float64, got {type(result)}"

    def test_mse_basic(self):
        predicted = np.array([1, 2, 3])
        observed = np.array([1, 2, 4])
        result = Distance.MSE(predicted, observed)
        assert np.isclose(result, 1/3), f"Expected 1/3, got {result}"

    def test_rmse_basic(self):
        predicted = np.array([1, 2, 3])
        observed = np.array([1, 2, 4])
        result = Distance.RMSE(predicted, observed)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_with_zeros(self):
        predicted = np.array([1, 2, 3, 4, 5])
        observed = predicted.copy()
        assert Distance.SSE(predicted, observed) == 0, "SSE with zeros should be zero"
        assert Distance.MSE(predicted, observed) == 0, "MSE with zeros should be zero"
        assert Distance.RMSE(predicted, observed) == 0, "RMSE with zeros should be zero"

    def test_with_negatives(self):
        predicted = np.array([-1, -2, -3])
        observed = np.array([-1, -2, -4])
        assert np.isclose(Distance.SSE(predicted, observed), 1.0), "SSE with negatives should be 1.0"
        assert np.isclose(Distance.MSE(predicted, observed), 1/3), "MSE with negatives should be 1/3"
        assert np.isclose(Distance.RMSE(predicted, observed), 1/np.sqrt(3)), "RMSE with negatives should be 1/sqrt(3)"

    def test_with_nan(self):
        predicted = np.array([1, np.nan, 3])
        observed = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            Distance.SSE(predicted, observed)
        with pytest.raises(ValueError):
            Distance.MSE(predicted, observed)
        with pytest.raises(ValueError):
            Distance.RMSE(predicted, observed)

    def test_with_inf(self):
        predicted = np.array([1, np.inf, 3])
        observed = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            Distance.SSE(predicted, observed)
        with pytest.raises(ValueError):
            Distance.MSE(predicted, observed)
        with pytest.raises(ValueError):
            Distance.RMSE(predicted, observed)

    def test_empty_arrays(self):
        predicted = np.array([])
        observed = np.array([])
        with pytest.raises(ValueError):
            Distance.SSE(predicted, observed)
        with pytest.raises(ValueError):
            Distance.MSE(predicted, observed)
        with pytest.raises(ValueError):
            Distance.RMSE(predicted, observed)

    def test_shape_mismatch(self):
        predicted = np.array([1, 2, 3])
        observed = np.array([1, 2])
        with pytest.raises(ValueError):
            Distance.SSE(predicted, observed)
        with pytest.raises(ValueError):
            Distance.MSE(predicted, observed)
        with pytest.raises(ValueError):
            Distance.RMSE(predicted, observed)

    def test_integer_and_float(self):
        predicted = np.array([1, 2, 3], dtype=int)
        observed = np.array([1.0, 2.0, 4.0], dtype=float)
        assert np.isclose(Distance.SSE(predicted, observed), 1.0), "Expected SSE to be 1.0"
        assert np.isclose(Distance.MSE(predicted, observed), 1/3), "Expected MSE to be 1/3"
        assert isinstance(Distance.RMSE(predicted, observed), float), "Expected RMSE to be a float"

    def test_2d_arrays(self):
        predicted = np.array([[1, 2], [3, 4]])
        observed = np.array([[1, 2], [3, 5]])
        assert np.isclose(Distance.SSE(predicted, observed), 1.0), "Expected SSE to be 1.0"
        assert np.isclose(Distance.MSE(predicted, observed), 0.25), "Expected MSE to be 0.25"
        assert np.isclose(Distance.RMSE(predicted, observed), 0.5), "Expected RMSE to be 0.5"

class TestDiscrete:
    def test_chisquare_basic(self):
        predicted = np.array([0.2, 0.3, 0.5])
        observed = np.array([2, 3, 5])
        result = Discrete.ChiSquare(predicted, observed)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert np.isclose(result, 0.0), "Expected ChiSquare to be 0.0 for perfect prediction"
        expected = np.float64(1.142857142857143)
        observed[2] = 9
        result = Discrete.ChiSquare(predicted, observed)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_g2_basic(self):
        predicted = np.array([0.2, 0.3, 0.5])
        observed = np.array([2, 3, 5])
        result = Discrete.G2(predicted, observed)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert np.isclose(result, 0.0), "Expected G2 to be 0.0 for perfect prediction"
        expected = np.float64(1.1589373428441814)
        observed[2] = 9
        result = Discrete.G2(predicted, observed)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_with_zeros(self):
        predicted = np.array([0.2, 0.3, 0.5])
        observed = np.zeros(3)
        result = Discrete.ChiSquare(predicted, observed)
        assert isinstance(result, float)
        result = Discrete.G2(predicted, observed)
        assert isinstance(result, float)

    def test_with_negatives(self):
        predicted = np.array([0.2, 0.3, 0.5])
        observed = np.array([-2, -3, -5])
        result = Discrete.ChiSquare(predicted, observed)
        assert isinstance(result, float)
        result = Discrete.G2(predicted, observed)
        assert isinstance(result, float)

    def test_with_nan(self):
        predicted = np.array([0.2, np.nan, 0.5])
        observed = np.array([2, 3, 5])
        with pytest.raises(ValueError):
            Discrete.ChiSquare(predicted, observed)
        with pytest.raises(ValueError):
            Discrete.G2(predicted, observed)

    def test_with_inf(self):
        predicted = np.array([0.2, np.inf, 0.5])
        observed = np.array([2, 3, 5])
        with pytest.raises(ValueError):
            Discrete.ChiSquare(predicted, observed)
        with pytest.raises(ValueError):
            Discrete.G2(predicted, observed)

    def test_empty_arrays(self):
        predicted = np.array([])
        observed = np.array([])
        with pytest.raises(ValueError):
            Discrete.ChiSquare(predicted, observed)
        with pytest.raises(ValueError):
            Discrete.G2(predicted, observed)

    def test_shape_mismatch(self):
        predicted = np.array([0.2, 0.3, 0.5])
        observed = np.array([2, 3])
        with pytest.raises(ValueError):
            Discrete.ChiSquare(predicted, observed)
        with pytest.raises(ValueError):
            Discrete.G2(predicted, observed)

    def test_integer_and_float(self):
        predicted = np.array([0.2, 0.3, 0.5], dtype=float)
        observed = np.array([2, 3, 5], dtype=int)
        result = Discrete.ChiSquare(predicted, observed)
        assert isinstance(result, float)
        result = Discrete.G2(predicted, observed)
        assert isinstance(result, float)

    def test_2d_arrays(self):
        predicted = np.array([[0.2, 0.3], [0.5, 0.0]])
        observed = np.array([[2, 3], [5, 0]])
        result = Discrete.ChiSquare(predicted, observed)
        assert isinstance(result, float)
        result = Discrete.G2(predicted, observed)
        assert isinstance(result, float)

if __name__ == "__main__":
    pytest.main()
