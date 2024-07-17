import numpy as np
import pandas as pd
import copy
import warnings
from ..generators.parameters import Parameters

__all__ = [
    "simulation_export",
    "unpack_trials",
    "unpack_participants",
    "determine_data_length",
    "cast_parameters",
]


def unpack_trials(data, i, pandas=True):
    """
    Unpack the data into a list of dictionaries.

    Parameters
    ----------
    data : pandas.DataFrame or dict
        A dataframe or dict containing the data to unpack.
    i : int
        The index of the data to unpack.
    pandas : bool
        Whether to return the data as a pandas dataframe.

    Returns
    -------
    list
        A list of dictionaries containing the data.
    """
    if pandas:
        trial = data.iloc[i, :].squeeze()
    else:
        trial = {k: data[k][i] for k in data.keys() if k != "ppt"}

    return trial


def unpack_participants(data, index, keys=None, pandas=True):
    """
    Unpack the data into a list of dictionaries.

    Parameters
    ----------
    data : pandas.DataFrameGroupBy or array_like
        A dataframe or list of dictionaries containing the data to unpack.

    Returns
    -------
    pd.DataFrame or dict
        A dataframe or dict containing a single participant's data.
    """
    if pandas and keys is not None:
        return data.get_group(keys[index])
    elif pandas and keys is None:
        return data.iloc[i:, :].squeeze()
    else:
        return data[index]


def determine_data_length(data):
    """
    This function determines the length of the data.

    Parameters
    ----------
    data : array_like or pandas.DataFrame
        The data to determine the length of.

    Returns
    -------
    int
        The length of the data.
    bool
        Whether the data is a pandas dataframe.
    """
    __pandas__ = True
    # find the shape of each key in the data
    if isinstance(data, dict):
        shape = [(np.array(v).shape) for k, v in data.items() if k != "ppt"]
        # find the maximum number of trials
        __len__ = np.max([shape[0] for shape in shape])
        __pandas__ = False
    if isinstance(data, pd.DataFrame):
        __len__ = len(data)
    return __len__, __pandas__


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
