import numpy as np
import pandas as pd
import copy

__all__ = [
    "simulation_export",
]


def simulation_export(simulation):
    """
    Return a pandas dataframe from the cpm-conventional array of dictionaries.

    Returns
    ------
    policies : pandas.DataFrame
        A dataframe containing the the model output for each participant and trial.
        If the output variable is organised as an array with more than one dimension, the output will be flattened.
    """
    simulation = copy.deepcopy(simulation)
    policies = pd.DataFrame()
    id = 0
    for i in simulation:
        ppt = pd.DataFrame()
        for k in i:
            row = pd.DataFrame()
            for key, value in k.items():
                if len(list(np.array(value).shape)) > 1:
                    Warning(
                        f"Value of {key} is of shape {value.shape}. It should be 1D."
                    )
                if isinstance(value, int) or isinstance(value, float):
                    value = np.array([value])
                current = pd.DataFrame(value.flatten()).T
                current.columns = [f"{key}_{i}" for i in range(current.shape[1])]
                row = pd.concat([row, current], axis=1)
            ppt = pd.concat([ppt, row], axis=0)
        ppt["ppt"] = id
        id += 1
        policies = pd.concat([policies, ppt.copy()], axis=0)
        del ppt
    policies = policies.reset_index(drop=True)
    return policies
