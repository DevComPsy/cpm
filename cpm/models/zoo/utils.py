import numpy as np

__all__ = ['embedding_nominal']

def embedding_nominal(stimuli = None, bits = None):
    """
    Converts a given array of stimuli into a binary representation.

    Parameters:
    stimuli (numpy.ndarray): The array of stimuli to be converted.

    Returns:
    numpy.ndarray: The binary representation of the stimuli.
    """
    output = np.zeros((bits))
    for i in stimuli:
        output[i - 1] = 1
    return output