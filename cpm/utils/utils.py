import pandas as pd


def PandasToDict(
    df, participant="ppt", stimuli="stimulus", feedback="feedback", **kwargs
):
    """
    Convert a pandas dataframe to a dictionary suitable for use with the CPM wrappers.

    The pandas dataframe should have a column for each stimulus, and a column for
    each feedback. Each row should be a single trial, and each participant should
    have a unique number in the participant column.

    Parameters
    ----------
    df : pandas dataframe
        The dataframe to convert
    stimuli : str, optional
        The prefix for each stimulus column in the pandas DataFrame, by default "stimulus".
    participant : str, optional
        The column name for the participant number, by default "ppt".
    **kwargs : dict, optional
        Any other keyword arguments to pass to the pandas DataFrame.

    Returns
    -------
    list
        A list of dictionaries, each dictionary containing the stimuli and
        feedback for a single participant.
    """
    names = df.columns.to_series()
    stimuli_indices = names[names.str.contains(stimuli)]
    length = df[participant].max()
    output = []

    for i in range(length):
        out = {}
        single = df[df[participant] == i + 1]
        out["stimuli"] = single[stimuli_indices].to_numpy()
        out["feedback"] = single.feedback.to_numpy()
        if kwargs is not None:
            for key, value in kwargs.items():
                out[key] = single[value].to_numpy()
        output.append(out.copy())

    return output


def DictToPandas(dict):
    # TODO: Add must handle multidimensional arrays per-column
    # TODO: You need to test it
    """
    Convert a dictionary to a pandas dataframe.

    Parameters
    ----------
    dict : dict
        The dictionary to convert.

    Returns
    -------
    pandas: dataframe
        The pandas dataframe converted from dict.
    """
    output = pd.DataFrame()
    for key, value in dict.items():
        if len(value.shape) > 1:
            for i in range(value.shape[1]):
                output[f"{key}_{i}"] = pd.Series(value[:, i].tolist())
        else:
            output[key] = pd.Series(value.tolist())
    return output
