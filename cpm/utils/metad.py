import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def count_trials(
    data=pd.DataFrame,
    stimuli: str = "Stimuli",
    responses: str = "Responses",
    accuracy: str = "Accuracy",
    confidence: str = "Confidence",
    nRatings: int = 4,
    padding: bool = False,
    padAmount: float = None,
):
    """Convert raw behavioral data to nR_S1 and nR_S2 response count.

    Given data from an experiment where an observer discriminates between two
    stimulus alternatives on every trial and provides confidence ratings,
    converts trial by trial experimental information for N trials into response
    counts.

    Parameters
    ----------
    data :
        Dataframe containing stimuli, accuracy and confidence ratings.
    stimuli :
        Stimuli ID (0 or 1). If a dataframe is provided, should be the name of
        the column containing the stimuli ID. Default is `'Stimuli'`.
    responses :
        Response (0 or 1). If a dataframe is provided, should be the
        name of the column containing the response accuracy. Default is
        `'Responses'`.
    accuracy :
        Response accuracy (0 or 1). If a dataframe is provided, should be the
        name of the column containing the response accuracy. Default is
        `'Accuracy'`.
    confidence :
        Confidence ratings. If a dataframe is provided, should be the name of
        the column containing the confidence ratings. Default is
        `'Confidence'`.
    nRatings :
        Total of available subjective ratings available for the subject. e.g.
        if subject can rate confidence on a scale of 1-4, then nRatings = 4.
        Default is `4`.
    padding :
        If `True`, each response count in the output has the value of padAmount
        added to it. Padding cells is desirable if trial counts of 0 interfere
        with model fitting. If False, trial counts are not manipulated and 0s
        may be present in the response count output. Default value for padding
        is 0.
    padAmount :
        The value to add to each response count if padding is set to 1.
        Default value is 1/(2*nRatings)

    Returns
    -------
    nR_S1, nR_S2 :
        Vectors containing the total number of responses in each accuracy
        category, conditional on presentation of S1 and S2.

    Notes
    -----
    All trials where `stimuli` is not 0 or 1, accuracy is not 0 or 1, or confidence is
    not in the range [1, nRatings], are automatically omitted.

    The inputs can be responses, accuracy or both. If both `responses` and
    `accuracy` are provided, will check for consstency. If only `accuracy` is
    provided, the responses vector will be automatically infered.

    If nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was presented, the subject had
    the following accuracy counts:
        responded S1, confidence=3 : 100 times
        responded S1, confidence=2 : 50 times
        responded S1, confidence=1 : 20 times
        responded S2, confidence=1 : 10 times
        responded S2, confidence=2 : 5 times
        responded S2, confidence=3 : 1 time

    The ordering of accuracy / confidence counts for S2 should be the same as it is for
    S1. e.g. if nR_S2 = [3 7 8 12 27 89], then when stimulus S2 was presented, the
    subject had the following accuracy counts:
        responded S1, confidence=3 : 3 times
        responded S1, confidence=2 : 7 times
        responded S1, confidence=1 : 8 times
        responded S2, confidence=1 : 12 times
        responded S2, confidence=2 : 27 times
        responded S2, confidence=3 : 89 times

    Examples
    --------
    >>> stimID = [0, 1, 0, 0, 1, 1, 1, 1]
    >>> accuracy = [0, 1, 1, 1, 0, 0, 1, 1]
    >>> confidence = [1, 2, 3, 4, 4, 3, 2, 1]
    >>> nRatings = 4

    >>> nR_S1, nR_S2 = trials2counts(stimID, accuracy, confidence, nRatings)
    >>> print(nR_S1, nR_S2)

    Reference
    ---------
    This function is adapted from the Python version of trials2counts.m by
    Maniscalco & Lau [1] retrieved at:
    http://www.columbia.edu/~bsm2105/type2sdt/trials2counts.py

    .. [1] Maniscalco, B., & Lau, H. (2012). A signal detection theoretic
        approach for estimating metacognitive sensitivity from confidence
        ratings. Consciousness and Cognition, 21(1), 422–430.
        https://doi.org/10.1016/j.concog.2011.09.021

    """
    if isinstance(data, pd.DataFrame):
        stimuli = data[stimuli].to_numpy()
        confidence = data[confidence].to_numpy()
        if accuracy in data:
            accuracy = data[accuracy].to_numpy()
        if responses in data:
            responses = data[responses].to_numpy()
    elif data is not None:
        raise ValueError("`Data` should be a DataFrame")

    # Check data consistency
    tempstim, tempresp, tempratg = [], [], []
    for s, rp, rt in zip(stimuli, responses, confidence):
        if (s == 0 or s == 1) and (rp == 0 or rp == 1) and (rt >= 1 and rt <= nRatings):
            tempstim.append(s)
            tempresp.append(rp)
            tempratg.append(rt)
    stimuli = tempstim
    responses = tempresp
    confidence = tempratg

    if padAmount is None:
        padAmount = 1 / (2 * nRatings)

    nR_S1, nR_S2 = [], []
    # S1 responses
    for r in range(nRatings, 0, -1):
        cs1, cs2 = 0, 0
        for s, rp, rt in zip(stimuli, responses, confidence):
            if s == 0 and rp == 0 and rt == r:
                cs1 += 1
            if s == 1 and rp == 0 and rt == r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)

    # S2 responses
    for r in range(1, nRatings + 1, 1):
        cs1, cs2 = 0, 0
        for s, rp, rt in zip(stimuli, responses, confidence):
            if s == 0 and rp == 1 and rt == r:
                cs1 += 1
            if s == 1 and rp == 1 and rt == r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)

    # pad response counts to avoid zeros
    if padding:
        nR_S1 = [n + padAmount for n in nR_S1]
        nR_S2 = [n + padAmount for n in nR_S2]

    return np.array(nR_S1), np.array(nR_S2)

def bin_ratings(
    ratings=None,
    nbins: int = 4,
    verbose: bool = True,
    ignore_invalid: bool = False,
):
    """Convert from continuous to discrete ratings.

    Resample if quantiles are equal at high or low end to ensure proper
    assignment of binned confidence

    Parameters
    ----------
    ratings : list | np.ndarray
        Ratings on a continuous scale.
    nbins : int
        The number of discrete ratings to resample. Defaut set to `4`.
    verbose : boolean
        If `True`, warning warnings be returned.
    ignore_invalid : bool
        If `False` (default), an arreor will be raised in case of impossible
        discretisation of the confidence ratings. This is mostly due to identical
        values and SDT values should not be extracted from the data. If `True` the
        discretisation will process anyway. This option can be usefull for plotting.

    Returns
    -------
    discreteRatings : np.ndarray
        New rating array only containing integers between 1 and `nbins`.
    out : dict
        Dictionary containing logs of the discrization process:
            * `'confbins'`: list or 1d array-like - If the ratings were
                reampled, a list containing the new ratings and the new low or
                hg threshold, appened before or after the rating, respectively.
                Else, only returns the ratings.
            * `'rebin'`: boolean - If True, the ratings were resampled due to
                larger numbers of highs or low ratings.
            * `'binCount'` : int - Number of bins

    .. warning:: This function will automatically control for bias in high or
        low confidence ratings. If the first two or the last two quantiles
        have identical values, low or high confidence trials are excluded
        (respectively), and the function is run again on the remaining data.

    Raises
    ------
    ValueError:
        If the confidence ratings contains a lot of identical values and
        `ignore_invalid` is `False`.

    Examples
    --------
    >>> from metadpy.utils import discreteRatings
    >>> ratings = np.array([
    >>>     96, 98, 95, 90, 32, 58, 77,  6, 78, 78, 62, 60, 38, 12,
    >>>     63, 18, 15, 13, 49, 26,  2, 38, 60, 23, 25, 39, 22, 33,
    >>>     32, 27, 40, 13, 35, 16, 35, 73, 50,  3, 40, 0, 34, 47,
    >>>     52,  0,  0,  0, 25,  1, 16, 37, 59, 20, 25, 23, 45, 22,
    >>>     28, 62, 61, 69, 20, 75, 10, 18, 61, 27, 63, 22, 54, 30,
    >>>     36, 66, 14,  2, 53, 58, 88, 23, 77, 54])
    >>> discreteRatings, out = discreteRatings(ratings)
    (array([4, 4, 4, 4, 2, 3, 4, 1, 4, 4, 4, 4, 3, 1, 4, 1, 1, 1, 3, 2, 1, 3,
        4, 2, 2, 3, 2, 2, 2, 2, 3, 1, 3, 1, 3, 4, 3, 1, 3, 1, 2, 3, 3, 1,
        1, 1, 2, 1, 1, 3, 3, 2, 2, 2, 3, 2, 2, 4, 4, 4, 2, 4, 1, 1, 4, 2,
        4, 2, 3, 2, 3, 4, 1, 1, 3, 3, 4, 2, 4, 3]),
    {'confBins': array([ 0., 20., 35., 60., 98.]), 'rebin': 0, 'binCount': 21})

    """
    out, temp = {}, []
    confBins = np.quantile(ratings, np.linspace(0, 1, nbins + 1))
    if (confBins[0] == confBins[1]) & (confBins[nbins - 1] == confBins[nbins]):
        if ignore_invalid is False:
            raise ValueError(
                "The resulting rating scale contains a lot of identical values and cannot be further analyzed."
                "Consider setting `ignore_invalid` to `True` to ignore this error."
                "This is not recommended for SDT analysis."
            )
    elif confBins[nbins - 1] == confBins[nbins]:
        if verbose is True:
            print("Correcting for bias in high confidence ratings")
        # Exclude high confidence trials and re-estimate
        hiConf = confBins[-1]
        confBins = np.quantile(ratings[ratings != hiConf], np.linspace(0, 1, nbins))
        for b in range(len(confBins) - 1):
            temp.append((ratings >= confBins[b]) & (ratings <= confBins[b + 1]))
        temp.append(ratings == hiConf)

        out["confBins"] = [confBins, hiConf]
        out["rebin"] = [1]
    elif confBins[0] == confBins[1]:
        if verbose is True:
            print("Correction for bias in low confidence ratings")
        # Exclude low confidence trials and re-estimate
        lowConf = confBins[1]
        temp.append(ratings == lowConf)
        confBins = np.quantile(ratings[ratings != lowConf], np.linspace(0, 1, nbins))
        for b in range(1, len(confBins)):
            temp.append((ratings >= confBins[b - 1]) & (ratings <= confBins[b]))
        out["confBins"] = [lowConf, confBins]
        out["rebin"] = [1]
    else:
        for b in range(len(confBins) - 1):
            temp.append((ratings >= confBins[b]) & (ratings <= confBins[b + 1]))
        out["confBins"] = confBins
        out["rebin"] = [0]

    discreteRatings = np.zeros(len(ratings), dtype="int")
    for b in range(nbins):
        discreteRatings[temp[b]] = b
    discreteRatings += 1
    out["binCount"] = [sum(temp[b])]

    return discreteRatings, out