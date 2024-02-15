import numpy as np


def Nominal(target=None, magnitude=None, bits=None):
    """
    Converts a target value to a binary representation using a specified number of bits.

    Parameters
    ----------
    target : int or numpy.ndarray
        The target value(s) to be converted.
    magnitude : float
        The magnitude of the binary representation, usually reward magnitude for cases with multiple possible outcomes.
    bits : int
        The number of possible values to use for determining the length of the binary representation.

    Returns
    -------
    numpy.ndarray:
        The binary representation of the target value(s).

    Raises
    ------
    ValueError:
        If the number of bits is less than the maximum stimulus value.
    ValueError:
        If the number of bits is 0 and the target value is less than 1.

    """
    output = np.zeros((bits))
    if np.max(target) > bits:
        raise ValueError(
            "The number of bits must be greater than or equal to the maximum stimulus value."
        )
    if bits == 0 and np.any(target < 1):
        raise ValueError(
            "The number of bits must be greater than or equal to the maximum stimulus value."
        )
    if np.any(target < 0) and bits == 1:
        # in case of negative values (negative reward magnitude)
        return target
    if np.any(target < 0) and bits > 1:
        # in case of negative values with multiple outcomes (negative reward magnitude)
        output[target - 1] = magnitude
    else:
        for i in target:
            output[i - 1] = 1
        return output
