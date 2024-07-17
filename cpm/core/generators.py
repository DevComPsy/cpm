import numpy as np
import pandas as pd

__all__ = [
    "extract_params_from_fit",
    "detailed_pandas_compiler",
    "generate_guesses",
]


def extract_params_from_fit(data, keys=None):
    """
    Extract the parameters from the fit.
    """
    parameters = {}
    for i in range(len(data)):
        parameters[keys[i]] = data[i]
    return parameters


def detailed_pandas_compiler(details):
    """
    Exports a list of dictionaries as a pandas dataframe.
    Optimised for the output of the routines implemented.

    Parameters
    ----------
    details : list
        A list of dictionaries containing the optimization details.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the optimization details.
    """
    output = pd.DataFrame()
    for i in details:
        row = pd.DataFrame()
        for key, value in i.items():
            if key == "population" or key == "population_energies":
                continue
            if isinstance(value, (list, np.ndarray)):
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    value = value.flatten()
                value = pd.DataFrame(np.asarray(value)).T
            else:
                value = pd.DataFrame([value]).T
            value.columns = [key + "_" + str(x) for x in value.columns]
            row = pd.concat([row, value], axis=1)
        output = pd.concat([output, row], axis=0)
    return output


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
