import numpy as np
import pandas as pd
import copy
import warnings
from ..generators.parameters import Parameters

__all__ = [
    "unpack_trials",
    "trial_reader",
    "unpack_participants",
    "determine_data_length",
    "extract_params_from_fit",
    "detailed_pandas_compiler",
    "decompose",
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


def trial_reader(data, pandas=True):
    """
    A function that returns trial `i` of `data`, as `unpack_trials(data, i, pandas)` does.

    Reading a row of a `pandas.DataFrame` with `iloc` takes about 50 microseconds,
    which a model run pays on every trial. For data with only numeric columns, or
    only boolean ones, the reader converts the data to a numpy array once and builds each row
    from it, which gives the same `pandas.Series` (values, dtype, index and name)
    about seven times faster. The array is a copy made when the reader is created,
    so the data are read afresh on every run, and changing a trial inside a model
    cannot change the data.

    Parameters
    ----------
    data : pandas.DataFrame or dict
        The data of one participant.
    pandas : bool
        Whether `data` is a `pandas.DataFrame`.

    Returns
    -------
    function
        A function of the trial index.
    """
    if pandas and isinstance(data, pd.DataFrame):
        dtypes = data.dtypes
        plain = all(
            isinstance(dtype, np.dtype) and dtype.kind in "biuf" for dtype in dtypes
        )
        values = data.to_numpy(copy=True) if plain and data.shape[1] > 0 else None
        ## mixed bool and numeric columns give an object array of Python scalars,
        ## where iloc gives numpy scalars, so those are read with iloc
        if values is not None and values.dtype != object:
            columns = data.columns
            index = data.index
            single = data.shape[1] == 1

            def read(i):
                row = pd.Series(values[i], index=columns, name=index[i])
                return row.squeeze() if single else row

            return read
    return lambda i: unpack_trials(data, i, pandas)


def unpack_participants(data, index, keys=None, pandas=True):
    """
    Return a single participant's data or parameter identified by an indexing variable.

    Parameters
    ----------
    data : pandas.DataFrame, pandas.api.texting.DataFrameGroupBy or array_like
        A pd.DataFrame, a grouped DataFrame or list of dictionaries containing the data to unpack.
    index : int
        The index of the data to unpack, or the index of the grouping variable if `keys` is not None.
    keys : array_like
        An array of keys according to which the data is grouped.
    pandas: bool
        Whether data is a pandas DataFrame or not.

    Returns
    -------
    pandas.DataFrame, pandas.Series, or dict
        A dataframe or dict containing a single participant's data. If the data are parameters, where each row is a participant, a pandas Series is returned.

    Notes
    -----
    If the data are parameters, the keys argument must be None.

    This function is mainly used in cpm.generators.Simulator to manage different data structures before updating and running the model on a data set with multiple participants.
    """
    if pandas and keys is not None:
        return data.get_group(keys[index])
    elif pandas and keys is None:
        return data.iloc[index]
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
    # find the shape of each key in the data
    if isinstance(data, dict):
        shape = [(np.array(v).shape) for k, v in data.items() if k != "ppt"]
        # find the maximum number of trials
        __len__ = np.max([shape[0] for shape in shape])
        __pandas__ = False
    if isinstance(data, pd.DataFrame):
        __len__ = len(data)
        __pandas__ = True

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
            ## ignore some items
            if key == "population" or key == "population_energies":
                continue
            ## make sure things are in the right format
            if isinstance(value, np.ndarray):
                value = value.flatten()
                ## ensure it is a 1D array
                value = value.reshape(-1)
            elif isinstance(value, list):
                value = np.asarray(value).flatten()
            elif isinstance(value, tuple):
                value = np.asarray(list(value)).flatten()
            elif isinstance(value, pd.Series):
                value = value.to_numpy().flatten()
            elif isinstance(value, dict):
                value = pd.DataFrame(value).T
            elif isinstance(value, (np.float64, np.int64, int, float, str, bool)):  #removed np.pool as this causes a bug with new numpy versions
                value = np.array([value])
            else:
                raise ValueError(
                    "The value of the optimiser output is not a numpy class, list, dict, or pandas dataframe. Please check the data.\n"
                    f"The value is of type {type(value)}.\n"
                    f"The value is {value}.\n"
                )
            value = pd.DataFrame(value).T
            value = value.reset_index(drop=True)
            
            ## rename the columns
            try: 
                if value.columns.shape[0] == 1:
                    value.columns = [key]
                else:
                    value.columns = [key + "_" + str(x) for x in value.columns]
            except AttributeError:
                raise ValueError(
                    "The value of the optimiser output is not a pandas dataframe. Please check the data.\n"
                    f"The value is of type {type(value)}.\n"
                    f"The value is {value}.\n"
                )
            row = pd.concat([row, value], axis=1)
        output = pd.concat([output, row], axis=0)
    return output



def decompose(participant=None, pandas=False, identifier=None):
    """
    Decompose the data for fitting. The function extracts subsets of data from the participant's data and returns them as separate variables.

    Parameters
    ----------
    participant : pandas.DataFrame or dict
        A dataframe or dict containing the data to unpack.
    pandas : bool
        Whether to return the data as a pandas dataframe.
    id : str
        The id of the participant.

    Returns
    -------
    participant : pandas.DataFrame or dict
        The participant data.
    observed : array_like
        The observed data to fit the model to.
    ppt : str or float
        The participant identifier.
    """
    if pandas:
        ppt, data = participant
        observed = data.observed.to_numpy()
    else:
        ppt = None
        data = participant
        observed = participant.get("observed")

    if identifier is not None and pandas is False:
        ppt = participant.get(identifier)

    return data, observed, ppt
