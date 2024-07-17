import numpy as np
import pandas as pd
import copy
import warnings
from ..generators.parameters import Parameters

__all__ = [
    "unpack_trials",
    "unpack_participants",
    "determine_data_length",
    "extract_params_from_fit",
    "detailed_pandas_compiler",
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
