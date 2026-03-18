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
    
    The data are from a risky decision-making task (similar to e.g., Rutledge et al., 2014), a two-alternative forced-choice task, which is implemented as a smartphone-compatible gamified application available in the BrainExplorer app (https://brainexplorer.net/) as the game called "_Scavenger_".
    On each trial, participants are presented with two stimuli and asked to select one of them. After they make their choice, they receive feedback in the form of rewards (coins).
    The stimuli are cartoonish depictions of stranded spaceships that participants have to salvage in order to obtain coins.
    In addition, participants are shown a radar, two circles drawn at the bottom of the screen, revealing the possible outcomes for each spaceship and their associated probabilities.
    One of the spaceships always returns an outcome with 100% probability (i.e., the safe option). The other spaceship has a 50% probability of returning an either a 0 or non-zero outcome (i.e., the risky option).
    In gain trials, all possible outcomes are positive (except the outcome of 0). In loss trials, all possible outcomes are negative (except the outcome of 0).
    Furthermore, forming an additional within-participant manipulation, trials can be either ambiguous or non-ambiguous.
    In ambiguous trials, the radar provides only partial information about the probabilities of the possible outcomes of the risky choice option.
    In these trials, the risky choice option is shown to have at least a 25% probability of returning an outcome of 0, and at least a 25% probability of returning a different outcome.
    Unbeknownst to the participant, the true probability of both outcomes is always 50/50 (which is why the risky_probability values in the dataset are always 0.5).
    Participants play one block with 40 trials, including 10 non-ambiguous gain trials, 10 non-ambiguous loss trials, 10 ambiguous gain trials, and 10 ambiguous loss trials.

    The dataset includes the following columns:

        - `ppt`: the participant number.
        - `safe_magnitudes`: the magnitude of the safe option.
        - `risky_magnitudes`: the non-zero outcome magnitude of the risky option.
        - `risky_probability`: the probability of the risky option.
        - `choice`: the choice of the participant (1 for risky, 0 for safe).
        - `optimal`: the optimal choice based on expected value (1 for risky, 0 for safe).
        - `feedback`: the feedback (reward) given to the participant.
        - `ambiguity`: whether the trial was ambiguous (1 for yes, 0 for no).
        - `attempt`: the number of the experimental session for a given participant (e.g. 2 here means that participant is completing the experiment the second time).
        - `rt_flip`: the reaction time for the choice, measured in seconds, multiplied by -1 if the choice is not an optimal choice. It is used for fitting various drift diffusion models.

    See Also
    --------

    [cpm.applications.decision_making.PTSM][cpm.applications.decision_making.PTSM]:  simplified version of the Prospect Theory-based Softmax Model (PTSM) for decision-making tasks based on Tversky & Kahneman (1992).

    [cpm.applications.decision_making.PTSM1992][cpm.applications.decision_making.PTSM1992]: full version of the Prospect Theory-based Softmax Model (PTSM) for decision-making tasks based on Tversky & Kahneman (1992).

    [cpm.applications.decision_making.PTSM2025][cpm.applications.decision_making.PTSM2025]: Prospect Theory Softmax Model loosely based on Chew et al. (2019).

    References
    ----------
    Rutledge, R. B., Skandali, N., Dayan, P., & Dolan, R. J. (2014). A computational and neural model of momentary subjective well-being. Proceedings of the National Academy of Sciences of the United States of America, 111(33), 12252–12257. https://doi.org/10.1073/pnas.1407535111

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
    
    The data are from an experiment combining a perceptual decision making task with an additional post-decision component assessing participants' confidence in their choices (Marzuki et al., 2025; Moses-Payne et al., 2021).
    The experiment is implemented as a gamified smartphone-compatible application available in the BrainExplorer platform (https://brainexplorer.net/) as the game "_Space Observer_".
    The task is designed to assess aspects of metacognition, i.e., the extent to which one can accurately reflect on and evaluate their own performance, specifically how well one’s confidence is calibrated to their actual performance.
    For a more thorough description of the experimental procedure, please refer to Marzuki et al. (2025).
    Briefly, in each trial of the game, participants are presented with a planet in the centre of the screen.
    On top of the planet, participants are shown 68 differently coloured aliens for 250 ms, overlaying the planet.
    The aliens are drawn from two distinct categories (S1 and S2), which differ in colour, but are identical in shape.
    After the 68 aliens disappear, participants are presented with an example of the two differently coloured aliens and asked to choose which of the two had been more abundant on that trial. 
    After each of the trials, they are asked to indicate how confident they are about their decision on a visual analogue scale ranging from “totally guessing” (0) to “totally certain” (100).
    The scale is not labelled with numbers and participants are not informed about the numerical values associated with their confidence ratings.
    A staircase procedure (Cornsweet, 1962; García-Pérez, 1998; Levitt, 1971) is used throughout the task to dynamically adjust the stimulus intensity, defined as the difference in aliens required to keep participants’ performance at approximately 70%.

    
    The dataset includes the following columns:

        - `participant`: the participant number.
        - `signal`: which alien is more abundant (0 = S1, 1 = S2).
        - `stimulus_intensity`: absolute difference in number of aliens (i.e., evidence strength). 
        - `response`: participant's choice (0 = S1, 1 = S2).
        - `confidence`: participant's confidence rating for their response.
        - `accuracy`: accuracy of the participant's response (0 = incorrect, 1 = correct).

    See Also
    --------
    [cpm.applications.signal_detection.EstimatorMetaD][cpm.applications.signal_detection.EstimatorMetaD]: Class to estimate metacognitive sensitivity (meta-d') from confidence ratings and choices.

    References
    ----------
    
    Chen, Y., Daly, H. R., Pitt, M. A., & Van Zandt, T. (2024). Assessing the distortions introduced when calculating d’: A simulation approach. _Behavior Research Methods_. https://doi.org/10.3758/s13428-024-02447-8
    
    Cornsweet, T. (1962). The staircase-method in psychophysics. Am. J. Psychol. 75, 485491. https://doi.org/10.2307/1419876
    
    García-Pérez M.A. (1998). Forced-choice staircases with fixed step sizes: asymptotic and small-sample properties. Vision Res., 38(12), 1861-81. https://doi.org/10.1016/s0042-6989(97)00340-4.
    
    Levitt, H. (1971). Transformed up-down methods in psychoacoustics. Journal of the Acoustical Society of America, 49, 467–477. https://doi.org/10.1121/1.1912375

    Marzuki, A., Kosina, L., Dome, L., Hewitt, S., & Hauser, T. (2025). Metacognitive antecedents to states of mental ill-health: Drops in confidence precede symptoms of OCD. _Research Square_. https://doi.org/10.21203/rs.3.rs-7544256/v1
    
    Moses‐Payne, M. E., Habicht, J., Bowler, A., Steinbeis, N., & Hauser, T. U. (2021). I know better! Emerging metacognition allows adolescents to ignore false advice. Developmental Science, 24(5), e13101. https://doi.org/10.1111/desc.13101


    """
    return load_csv("metacognition_data.csv")