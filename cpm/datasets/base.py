import pandas as pd
import os


def load_csv(filename):
    """
    Load a CSV file from the data module.

    Parameters
    ----------
    filename : str
        The name of the CSV file to load.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data from the CSV file.
    """
    # Construct the absolute file path
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    filepath = os.path.join(data_dir, filename)

    # Check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file or directory: '{filepath}'")

    data = pd.read_csv(filepath)
    return data


def load_bandit_data():
    """
    Load the 2-arm bandit dataset from a CSV file.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the 4-arm bandit data.

    Examples
    --------
    >>> data = load_bandit_data()
    >>> print(data.head())

    Notes
    -----
    The dataset will include the following columns:

        - `index`: variable to identify each row - this variable is clutter.
        - `ppt`: the participant number.
        - `trial`: the trial number.
        - `arm_left`: the stimulus presented on the left side.
        - `arm_right`: the stimulus presented on the right side.
        - `reward_left`: the reward received when the left stimulus is selected.
        - `reward_right`: the reward received when the right stimulus is selected.
        - `response`: the response of the participant (1 for right, 0 for left).
        - `feedback`: the feedback received by the participant (1 for reward, 0 for no reward).
    """
    return load_csv("bandit_small.csv")


# def load_another_dataset():
#     """
#     Load another dataset from a CSV file.

#     Returns
#     -------
#     pd.DataFrame
#         A pandas DataFrame containing the data from the CSV file.
#     """
#     return load_csv("another_dataset.csv")


# Add more functions for other CSV files as needed
