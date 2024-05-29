import numpy as np
import pandas as pd

__all__ = ["extract_params_from_fit", "detailed_pandas_compiler"]


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
    method : str
        The method used for the optimization. Default is 'differential_evolution'.

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
            if isinstance(value, list) or isinstance(value, np.ndarray):
                value = pd.DataFrame(np.asarray(value)).T
            else:
                value = pd.DataFrame([value]).T
            value.columns = [key + "_" + str(x) for x in value.columns]
            row = pd.concat([row, value], axis=1)
        output = pd.concat([output, row], axis=0)
    return output
