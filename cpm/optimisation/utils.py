import numpy as np
import pandas as pd

def ConvertToDict(data):
    """
    Convert the objects into a dictionary.
    """
    data_dict = {}
    for key, value in data.items():
        data_dict[key] = value
    return data_dict

def ExtractParamsFromFit(data, keys = None):
    """
    Extract the parameters from the fit.
    """
    parameters = {}
    for i in range(len(data)):
        parameters[keys[i]] = data[i]
    return parameters

def ConvertToPandas(a):
    m, n, r = a.shape
    out_arr = np.column_stack((np.repeat(np.arange(m), n), a.reshape(m*n, -1)))
    out_df = pd.DataFrame(out_arr)
    return out_df
