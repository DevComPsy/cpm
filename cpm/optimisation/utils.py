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