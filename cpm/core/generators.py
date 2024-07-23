import numpy as np
import copy
import pandas as pd
import warnings
from ..generators.parameters import Parameters

__all__ = [
    "cast_parameters",
    "generate_guesses",
]


def generate_guesses(
    bounds,
    number_of_starts=None,
    guesses=None,
    shape=None,
):
    """
    The function generates initial guesses for the optimization routine.

    Parameters
    ----------
    bounds : tuple
        The bounds of the parameters as output by `cpm.generators.Parameters.bounds()`.
    number_of_starts : int
        The number of initial guesses to generate.
    guesses : list
        A list of initial guesses. If provided, the function will use these guesses instead of generating new ones.
    shape : tuple
        The shape of the array of initial guesses.

    Returns
    -------
    np.ndarray
        An array of initial guesses.

    Notes
    -----
    If any of the bounds is `np.inf`, the function will generate guesses from an exponential distribution.
    If any of the bounds is other than `np.inf` or a finite number, the function will generate guesses from a normal distribution with a mean of 0 and sd of 1.
    """

    low, high = bounds[0], bounds[1]

    if number_of_starts is not None and guesses is not None:
        ## convert to a 2D array
        guesses = np.asarray(guesses)
        if len(guesses.shape) == 1:
            guesses = np.expand_dims(guesses, axis=0)
            ## assign the initial guess and raise an error if the number of starts does not match the number of initial guesses
            if np.asarray(guesses).shape[0] != number_of_starts:
                raise ValueError(
                    "The number of initial guesses must match the number of starts."
                )

    if number_of_starts is not None and guesses is None:

        guesses = np.empty(shape)

        for i in range(shape[1]):

            low, high = bounds[0][i], bounds[1][i]

            if np.isfinite(low) and np.isfinite(high):
                guesses[:, i] = np.random.uniform(low, high, shape[0])

            elif np.isfinite(low) and np.isinf(high):
                guesses[:, i] = low + np.random.exponential(scale=10, size=shape[0])
            elif np.isinf(low) and np.isfinite(high):
                guesses[:, i] = high - np.random.exponential(scale=10, size=shape[0])
            else:
                guesses[:, i] = np.random.normal(
                    loc=0, scale=1, size=shape[0]
                )  # Adjust the loc and scale as needed

    return guesses


def cast_parameters(parameters, sample=None):
    """
    Identify parameter type and repeat it for each participant.

    Parameters
    ----------
    parameters : dict, list, pd.Series, pd.DataFrame or cpm.generators.Parameters
        The parameters to cast.
    """
    cast = len(parameters) != sample
    if cast:
        if isinstance(parameters, dict):
            output = [copy.deepcopy(parameters) for i in range(1, sample + 1)]
        if isinstance(parameters, pd.Series):
            output = pd.DataFrame([parameters for i in range(1, sample + 1)])
        if isinstance(parameters, pd.DataFrame):
            repeats = sample // len(
                parameters
            )  # Calculate how many times to repeat the DataFrame to fit into sample
            remainder = sample % len(
                parameters
            )  # Calculate the remainder to adjust the final DataFrame size
            output = pd.concat(
                [parameters] * repeats + [parameters.iloc[:remainder]],
                ignore_index=True,
            )
        if isinstance(parameters, list):
            output = [copy.deepcopy(parameters) for i in range(1, sample + 1)]
        if isinstance(parameters, Parameters):
            output = parameters.sample(sample)
        warnings.warn(
            "The number of parameter sets and number of participants in data do not match.\nUsing the same parameters for all participants."
        )
    else:
        output = parameters

    return output
