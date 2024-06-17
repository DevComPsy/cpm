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
        print("success")


if __name__ == "__main__":
    pytest.main()
