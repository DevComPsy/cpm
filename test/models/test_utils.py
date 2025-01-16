import pytest
import numpy as np
from cpm.models.utils import Nominal


def test_nominal_positive_values():
    target = np.array([3, 1, 2])
    bits = 4
    expected_output = np.array([1, 1, 1, 0])
    assert np.array_equal(Nominal(target=target, bits=bits), expected_output)


def test_nominal_negative_values():
    target = np.array([-1, -2, -3])
    bits = 1
    expected_output = np.array([-1, -2, -3])
    assert np.array_equal(Nominal(target=target, bits=bits), expected_output)


def test_nominal_magnitude():
    target = np.array([1, 2])
    magnitude = 0.5
    bits = 3
    expected_output = np.array([0.5, 0.5, 0])
    assert np.array_equal(
        Nominal(target=target, magnitude=magnitude, bits=bits), expected_output
    )


def test_nominal_invalid_bits():
    target = np.array([1, 2, 3])
    bits = 2
    with pytest.raises(ValueError):
        Nominal(target=target, bits=bits)


def test_nominal_zero_bits():
    target = np.array([0])
    bits = 0
    with pytest.raises(ValueError):
        Nominal(target=target, bits=bits)


def test_nominal_large_bits():
    target = np.array([1, 2, 3])
    bits = 100
    expected_output = np.zeros(bits)
    expected_output[:3] = 1
    assert np.array_equal(Nominal(target=target, bits=bits), expected_output)


if __name__ == "__main__":
    pytest.main()
