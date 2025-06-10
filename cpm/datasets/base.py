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
        - `left`: the stimulus presented on the left side.
        - `right`: the stimulus presented on the right side.
        - `reward_left`: the reward received when the left stimulus is selected.
        - `reward_right`: the reward received when the right stimulus is selected.
        - `ppt`: the participant number.
        - `responses`: the response of the participant (1 for right, 0 for left).
    """
    return load_csv("bandit_small.csv")


def load_risky_choices():
    """
    Load the risky choices dataset from a CSV file.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the risky choices data.

    Examples
    --------
    >>> data = load_risky_choices()
    >>> print(data.head())

    Notes
    -----
    The dataset will include the following columns:

        - `ppt`: the participant number.
        - `safe_magnitudes`: the magnitudes of the safe option.
        - `risky_magnitudes_1`, `risky_magnitudes_2`: the magnitudes of the risky options.
        - `safe_prob`: the probability of the safe option.
        - `risky_prob_1`, `risky_prob_2`: the probabilities of the risky options.
        - `choice`: the choice of the participant (1 for risky, 0 for safe).
        - `chose_left`: whether the participant chose the left option (1 for left, 0 for right).

    """
    return load_csv("risky_choices.csv")

def load_model_based_model_free():
    """
    Load the risky choices dataset from a CSV file.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the risky choices data.

    Examples
    --------
    >>> data = load_model_based_model_free()
    >>> print(data.head())

    Notes
    -----
    The dataset will include the following columns:

        - userID: unique identifier for each participant
        - stimuli: which planets presented in stage of the trial
        - s: the state of the trial
        - rews: the rewards for each stimulus / planet 
        - choice: the choice made by the participant
        - stake: the stake of the trial (1 or 5) 
        - points: the points received in the trial

    """
    return load_csv("model_based_model_free.csv")
