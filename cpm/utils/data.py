import pandas as pd
import pickle as pkl

__all__ = ["pandas_to_dict", "dict_to_pandas", "load"]


def pandas_to_dict(
    df=None, participant="ppt", stimuli="stimulus", feedback="feedback", **kwargs
):
    """
    Convert a pandas dataframe to a dictionary suitable for use with the CPM wrappers.

    The pandas dataframe should have a column for each stimulus, and a column for
    each feedback, and a column for a participant identifier. Each row should be
    a single trial, and each participant should have a unique number in the participant column.

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
    feedback_indices = names[names.str.contains(feedback)]
    participants = pd.unique(df[participant])
    output = []

    for i in participants:
        out = {}
        single = df[df[participant] == i]
        out["trials"] = single[stimuli_indices].to_numpy()
        out["feedback"] = single[feedback_indices].to_numpy()
        out["ppt"] = pd.unique(single[participant])[0]
        if kwargs is not None:
            for key, value in kwargs.items():
                out[key] = single[value].to_numpy()
        output.append(out.copy())

    return output


def dict_to_pandas(dict):
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


def combine_lists(**kwargs):
    """
    This is an internally used utility function to combine multiple lists into a single list of lists, which can be used for data manipulation, such as in the convert_to_RLRW function. It takes in keyword arguments where each key is a type of data with their corresponding column names (>= 1). The function also handles the case where a single column name is provided as a string instead of a list, by converting it into a list with one element. This ensures that the output is always a list of lists, regardless of the input format.

    Parameters
    ----------
    kwargs : dict
        A dictionary of keyword arguments where each key is a type of data (e.g., 'stimulus', 'human_response', 'feedback') and the value is a list of column names corresponding to that type of data in the input dataframe.

    Returns
    -------
    dict
        A dictionary where each key is a type of data and the value is a list of column names for that type of data, in the order they were provided in the keyword arguments.
    """
    return {
        key: value if isinstance(value, list) else [value]
        for key, value in kwargs.items()
    }


def convert_to_RLRW(data, human_response, reward, stimulus, participant, **kwargs):
    """
    Convert a dictionary to a pandas dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        The dictionary to convert.
    human_response : str or list
        The column name for the human response.
    reward : str or list
        The column name for the reward.
    stimulus : str or list
        The column name(s) for the stimulus.
    participant : str
        The column name for the participant identifier.
    kwargs : dict, optional
        Any other keyword arguments to pass to the pandas DataFrame.

    Returns
    -------
    pandas.DataFrame
        The pandas.DataFrame compatible with the RLRW wrapper, containing the stimulus, human response, and reward columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from cpm.utils.data import convert_to_RLRW
    >>> data = pd.DataFrame({
    ...     "participant_id": [1, 1, 2],
    ...     "stim_left": [0, 1, 0],
    ...     "stim_right": [1, 0, 1],
    ...     "choice": [1, 0, 1],
    ...     "reward_left": [1, 0, 1],
    ...     "reward_right": [0, 1, 0],
    ...     "block": [1, 1, 2],
            "condition": ["A", "A", "B"]
    ... })
    >>> output = convert_to_RLRW(
    ...     data=data,
    ...     human_response="choice",
    ...     reward=["reward_left", "reward_right"],
    ...     stimulus=["stim_left", "stim_right"],
    ...     participant="participant_id",
    ...     block="block",
    ...     condition="condition"
    ... )
    >>> print(output)
       arm_0  arm_1  response  reward_0  reward_1  participant  block condition
    0       0       1         1         1         0            1      1         A
    1       1       0         0         0         1            1      1         A
    2       0       1         1         1         0            2      2         B
    """
    ## perform a series of checks to ensure that the specified columns exist in the input data
    if participant not in data.columns:
        raise ValueError(f"Participant column '{participant}' not found in data.")
    if isinstance(stimulus, str):
        if stimulus not in data.columns:
            raise ValueError(f"Stimulus column '{stimulus}' not found in data.")
    elif isinstance(stimulus, list):
        for stim in stimulus:
            if stim not in data.columns:
                raise ValueError(f"Stimulus column '{stim}' not found in data.")
    else:
        raise ValueError("Stimulus must be a string or a list of strings.")
    if isinstance(human_response, list):
        for resp in human_response:
            if resp not in data.columns:
                raise ValueError(f"Human response column '{resp}' not found in data.")
    elif isinstance(human_response, str):
        if human_response not in data.columns:
            raise ValueError(
                f"Human response column '{human_response}' not found in data."
            )
    else:
        raise ValueError("Human response must be a string or a list of strings.")
    if isinstance(reward, list):
        for rew in reward:
            if rew not in data.columns:
                raise ValueError(f"Reward column '{rew}' not found in data.")
    elif isinstance(reward, str):
        if reward not in data.columns:
            raise ValueError(f"Reward column '{reward}' not found in data.")
    else:
        raise ValueError("Reward must be a string or a list of strings.")
    ## check kwargs for any additional columns that need to be included in the output dataframe
    if kwargs is not None:
        for key, value in kwargs.items():
            if isinstance(value, list):
                for col in value:
                    if col not in data.columns:
                        raise ValueError(
                            f"Additional column '{col}' not found in data."
                        )
            elif isinstance(value, str):
                if value not in data.columns:
                    raise ValueError(f"Additional column '{value}' not found in data.")
            else:
                raise ValueError(
                    "Additional column names must be strings or lists of strings."
                )
    output = pd.DataFrame()
    ## combine column names for stimulus and human response into a dictionary of lists
    combined_columns = combine_lists(
        arm=stimulus,
        response=human_response,
        reward=reward,
        participant=participant,
        **kwargs,
    )
    for key, cols in combined_columns.items():
        ## create new colum names that match in length with the number of columns in the input data for that type of data
        length = len(cols)
        for i, col in enumerate(cols):
            new_col_name = f"{key}_{i}" if length > 1 else key
            output[new_col_name] = data[col]
    return output
