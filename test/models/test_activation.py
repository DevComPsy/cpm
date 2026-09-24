import pytest
import numpy as np
from cpm.models.activation import (
    Offset,
    SigmoidActivation,
    ProspectUtility,
    CompetitiveGating,
)


class TestOffset:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.vals = np.array([2.1, 1.1])
        self.offsetter = Offset(input=self.vals, offset=1.33, index=0)

    def test_compute(self):
        result = self.offsetter.compute()
        expected = np.array([3.43, 1.1])
        np.testing.assert_array_equal(result, expected)


class TestSigmoidActivation:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.weights = np.array([2.1, 1.1])
        self.input = np.array([1, 0])
        self.sigmoid = SigmoidActivation(input=self.input, weights=self.weights)

    def test_compute(self):
        result = self.sigmoid.compute()
        expected = np.array([0.890903, 0.5])
        np.testing.assert_array_almost_equal(result, expected)


class TestProspectUtility:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.vals = np.array([np.array([1, 40]), np.array([10])], dtype=object)
        self.probs = probs = np.array(
            [np.array([0.95, 0.05]), np.array([1])], dtype=object
        )
        self.prospect_utility = ProspectUtility(
            magnitudes=self.vals, probabilities=self.probs, alpha=0.85, gamma=0.9
        )

    def test_compute(self):
        result = self.prospect_utility.compute()
        # Replace 'expected' with the expected output of the compute method
        expected = np.array([2.44583162, 7.07945784])
        np.testing.assert_array_almost_equal(result, expected)


class TestCompetitiveGating:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.input = np.array([1, 1, 0])
        self.values = np.array([[0.1, 0.9, 0.8], [0.6, 0.2, 0.1]])
        self.salience = np.array([0.1, 0.2, 0.3])
        self.competitive_gating = CompetitiveGating(
            self.input, self.values, self.salience, P=1
        )

    def test_compute(self):
        result = self.competitive_gating.compute()
        expected = np.array([[0.03333333, 0.6, 0.0], [0.2, 0.13333333, 0.0]])
        np.testing.assert_array_equal(
            result.flatten().round(6), expected.flatten().round(6)
        )


@pytest.mark.parametrize("weighting", ["tk", "power", "prelec", "gw"])
def test_prospect_utility_on_several_outcomes_per_option(weighting):
    """Options with several outcomes, as 2D arrays; `prelec` used to fail on them."""
    from cpm.models import kernels

    rng = np.random.default_rng(1)
    magnitudes = rng.normal(0, 1, (3, 4))
    probabilities = rng.uniform(0.01, 1, (3, 4))
    model = ProspectUtility(magnitudes=magnitudes, probabilities=probabilities, alpha=0.8, beta=0.9,
                            lambda_loss=1.7, gamma=0.6, delta=0.8, weighting=weighting)
    got = model.compute()
    code = kernels.WEIGHTING[weighting]
    expected = [
        sum(kernels.prospect_weight(p, m, 0.6, 0.8, code) * kernels.prospect_utility(m, 0.8, 0.9, 1.7)
            for m, p in zip(magnitudes[j], probabilities[j]))
        for j in range(3)
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


if __name__ == "__main__":
    pytest.main()
