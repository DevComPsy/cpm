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
    
    The data are from a risky decision-making task called "Scavenger". On each trial, participants are presented with two stranded spaceships, and are instructed to salvage one of the two ships and observe the outcome of this action. 
    They have a radar showing them what the possible outcomes are for each spaceship, and what the probabilities are of obtaining these outcomes.
    One of the spaceships always returns an outcome with 100% probability (i.e., the safe option). The other spaceship has a 50% probability of returning an outcome of 0, and a 50% probability of returning a different outcome (i.e., the risky option). 
    After selecting one of the two options, participants receive feedback on the resulting outcome. 
    In gain trials, all possible outcomes are positive (except the outcome of 0). In loss trials, all possible outcomes are negative (except the outcome of 0).  
    In ambiguous trials, the radar provides only partial information about the probabilities of the possible outcomes of the risky choice option. In these trials, the risky choice option is shown to have at least a 25% probability of returning an outcome of 0, and at least a 25% probability of returning a different outcome. 
    Unbeknownst to the participant, the true probability of both outcomes is always 50/50 (which is why the risky_probability values in the dataset are always 0.5).  
    Participants play one block with 40 trials, including 10 non-ambiguous gain trials, 10 non-ambiguous loss trials, 10 ambiguous gain trials, and 10 ambiguous loss trials.  

    
    The dataset will include the following columns:

        - `ppt`: the participant number.
        - `safe_magnitudes`: the magnitudes of the safe option.
        - `risky_magnitudes`: the magnitudes of the risky options.
        - `risky_probability`: the probabilities of the risky options.
        - `choice`: the choice of the participant (1 for risky, 0 for safe).
        - `optimal`: the optimal choice based on expected value (1 for risky, 0 for safe).
        - `feedback`: the feedback (reward) given to the participant.
        - `ambiguity`: whether the choice was ambiguous (1 for yes, 0 for no).
        - `attempt`: the number of the experimental session for a given participant (e.g. 2 here means that participant is completing the experiment the second time).
        - `rt_flip`: the reaction time for the choice, measured in seconds, multiplied by -1 if the choice is not an optimal choice. It is used for fitting various drift diffusion models.

    """
    return load_csv("risky_choices.csv")


def load_metacognition_data():
    """
    Load the metacognition dataset from a CSV file.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the metacognition data.

    Examples
    --------
    >>> data = load_metacognition_data()
    >>> print(data.head())

    Notes
    -----
    
    The data are from a perceptual metacognitive task called "Space Observer" (see Marzuki et al., 2025; Moses-Payne et al., 2021). The task assesses the ability to accurately reflect on and evaluate one’s own cognitive processes, known as metacognition.  
    In each trial of the game, participants are presented with a planet in the centre of the screen. For a very brief time window (250 ms), 68 aliens appear into view, overlaying the planet. 
    Aliens can be one of eight colours; two colours are selected at random for each trial. Aliens are identical except for their colour. 
    After the 68 aliens disappear, participants are presented with an example of the two aliens and asked to choose which of the two had been more abundant on that trial. 
    After each of the trials, they are asked to indicate how confident they are about their decision on a visual analogue scale ranging from “totally guessing” (0) to “totally certain” (100). 
    A staircase procedure is used throughout the task to determine the stimulus intensity or evidence strength, defined as the difference in aliens required to keep participants’ performance at approximately 70%. 

    
    The dataset will include the following columns:

        - `participant`: the participant number.
        - `signal`: which alien is more abundant (0 = S1, 1 = S2).
        - `stimulus_intensity`: absolute difference in number of aliens (i.e., evidence strength). 
        - `response`: participant's choice (0 = S1, 1 = S2).
        - `confidence`: participant's confidence rating for their response.
        - `accuracy`: accuracy of the participant's response (0 = incorrect, 1 = 1 correct).

    """
    return load_csv("metacognition_data.csv")