import numpy as np
import pandas as pd
from typing import Union


def bin_ratings(
    ratings: Union[list, np.ndarray],
    nbins: int = 4,
    ignore_invalid: bool = False,
    xtdout: bool = False,
):
    """
    Convert continuous confidence ratings into discrete bins defined by quantiles.

    Resample if quantiles are equal at high or low end to ensure proper
    assignment of binned confidences.

    Parameters
    ----------
    ratings : list | np.ndarray
        Ratings on a continuous scale.
    nbins : int
        The number of discrete ratings to resample. Defaut set to `4`.
    ignore_invalid : bool
        If `False` (default), an ValueError will be raised in case of impossible
        discretisation of the confidence ratings. This is mostly due to identical
        values and SDT values should not be extracted from the data. If `True` the
        discretisation will process anyway.

    Returns
    -------
    discrete_ratings : np.ndarray
        If xtdout=False, the function returns the discretised ratings.
    xtdout : pd.Series
        if xtdout=True, the function returns a pd.Series with the following items:
        discrete_ratings : np.ndarray
            The discretised ratings.
        confidence_bins : np.ndarray
            The bins used for discretisation.
        resampled : bool
            If resampling was needed, it can either be True or False.
        counts_of_bins : int
            The number of bins.

    Raises
    ------
    ValueError:
        If the confidence ratings contains a lot of identical values and
        `ignore_invalid` is `False`.

    Examples
    --------
    >>> from cpm.utils.signal import discrete_ratings
    >>> ratings = np.array([
    >>>     96, 98, 95, 90, 32, 58, 77,  6, 78, 78, 62, 60, 38, 12,
    >>>     63, 18, 15, 13, 49, 26,  2, 38, 60, 23, 25, 39, 22, 33,
    >>>     32, 27, 40, 13, 35, 16, 35, 73, 50,  3, 40, 0, 34, 47,
    >>>     52,  0,  0,  0, 25,  1, 16, 37, 59, 20, 25, 23, 45, 22,
    >>>     28, 62, 61, 69, 20, 75, 10, 18, 61, 27, 63, 22, 54, 30,
    >>>     36, 66, 14,  2, 53, 58, 88, 23, 77, 54])
    >>> discrete_ratings(ratings)
    array([4, 4, 3, ..., 1, 1, 2])
    >>> discrete_ratings, out = bin_ratings(ratings, xtdout=True)
    confidence_bins     [0.00011738282511786213, 0.2518697968335193, 0...
    resampled                                                       False
    counts_of_bins                                                 [2500]
    discrete_ratings    [4, 4, 3, 1, 2, 1, 2, 3, 4, 4, 2, 1, 4, 1, 3, ...

    """
    out, temp = {}, []
    confidence_bins = np.quantile(ratings, np.linspace(0, 1, nbins + 1))
    if (confidence_bins[0] == confidence_bins[1]) & (
        confidence_bins[nbins - 1] == confidence_bins[nbins]
    ):
        if ignore_invalid is False:
            raise ValueError(
                "The resulting rating scale contains a lot of identical"
                " values and cannot be further analyzed"
            )
    elif confidence_bins[nbins - 1] == confidence_bins[nbins]:
        # Exclude high confidence trials and re-estimate
        high_confidence = confidence_bins[-1]
        confidence_bins = np.quantile(
            ratings[ratings != high_confidence], np.linspace(0, 1, nbins)
        )
        for b in range(len(confidence_bins) - 1):
            temp.append(
                (ratings >= confidence_bins[b]) & (ratings <= confidence_bins[b + 1])
            )
        temp.append(ratings == high_confidence)

        out["confidence_bins"] = [confidence_bins, high_confidence]
        out["resampled"] = [1]
    elif confidence_bins[0] == confidence_bins[1]:
        # Exclude low confidence trials and re-estimate
        low_confidence = confidence_bins[1]
        temp.append(ratings == low_confidence)
        confidence_bins = np.quantile(
            ratings[ratings != low_confidence], np.linspace(0, 1, nbins)
        )
        for b in range(1, len(confidence_bins)):
            temp.append(
                (ratings >= confidence_bins[b - 1]) & (ratings <= confidence_bins[b])
            )
        out["confidence_bins"] = [low_confidence, confidence_bins]
        out["resampled"] = True
    else:
        for b in range(len(confidence_bins) - 1):
            temp.append(
                (ratings >= confidence_bins[b]) & (ratings <= confidence_bins[b + 1])
            )
        out["confidence_bins"] = confidence_bins
        out["resampled"] = False

    discrete_ratings = np.zeros(len(ratings), dtype="int")
    for b in range(nbins):
        discrete_ratings[temp[b]] = b
    discrete_ratings += 1
    out["counts_of_bins"] = [sum(temp[b])]
    out["discrete_ratings"] = discrete_ratings

    if xtdout:
        return pd.Series(out)
    else:
        return discrete_ratings


def count_bins(
    stimulus_identifier, response, rating, n_ratings, padding=True, pandas=True
):
    """
    Convert discretized confidence ratings (usually sorted into bins) into response counts for each stimulus type.

    Parameters
    ----------
    stimulus_identifier : list-like
        List of stimulus identifiers. Must start from 0, so that if you have 4 stimuli, the identifiers will be [0, 1, 2, 3].
    response : list-like
        List of responses, where each element corresponds to one of the stimulus identifiers. Must start from 0.
    rating : list-like
        List of confidence ratings, where each element corresponds to elements in stimulus_identifier.
    n_ratings : int
        The number of discrete ratings, must be larger than 0 and smaller or equal to the highest value in rating.
    padding : bool or double
        If True, padding is added to avoid zeros in the response counts. The default value is 1 / (2 * n_ratings).
        If False, no padding is added. If a double is provided, it is used as padding.

    Returns
    -------
    counts : np.ndarray
        A numpy array with the response counts for each stimulus type and for each ratings. Each row is a stimulus type and each column is a rating, such that stimulus type 0, rating 2 with response 2 is counts[0, 2 * n_ratings + 2]. Note that counting starts from 0.

    Examples
    -------
    >>> import numpy as np
    >>> from cpm.utils.metad import discrete_ratings, trials2counts
    >>> n = 1000
    >>> dimensions = 4
    >>> stimulus_identifier = np.random.randint(0, dimensions, n)
    >>> response = np.random.randint(0, dimensions, n)
    >>> rating = np.random.normal(0, 1, n)
    >>> discrete = bin_ratings(rating, nbins=4)
    >>> n_ratings = 4
    >>> count_bins(stimulus_identifier=stimulus_identifier, response=response, rating=discrete, n_ratings=n_ratings, padding=False)

    """
    ## check for valid inputs, and force everything to numpy.array
    stimulus_identifier = np.array(stimulus_identifier)
    response = np.array(response)
    rating = np.array(rating)

    ## number of distinct responses
    n_responses = response.max() + 1
    if isinstance(padding, bool):
        if padding == True:
            padding = 1 / (2 * n_ratings)
    # check for valid inputs
    if not (len(stimulus_identifier) == len(response)) or not (
        len(stimulus_identifier) == len(rating)
    ):
        raise ValueError(
            "stimulus_identifier, response, and rating input vectors must have the same lengths"
        )

    number_of_stimuli = stimulus_identifier.max() + 1
    number_of_discrete_ratings = n_ratings
    columns = number_of_discrete_ratings * number_of_stimuli

    counts = np.zeros((number_of_stimuli, columns))

    for stim in np.arange(number_of_stimuli):
        indices = np.where(stimulus_identifier == stim)
        stim_ratings = rating[indices]
        stim_responses = response[indices]
        for resp in np.arange(n_responses):
            response_indices = np.where(stim_responses == resp)
            response_ratings = stim_ratings[response_indices]
            response_counts = np.zeros(n_ratings)
            response_unique = np.unique(response_ratings, return_counts=True)
            for rates in np.arange(len(response_unique[0])):
                response_counts[int(response_unique[0][rates]) - 1] = response_unique[
                    1
                ][rates]
            counts[stim, resp * n_ratings : (resp + 1) * n_ratings] = response_counts
    # pad response counts to avoid zeros
    if padding > 0:
        counts += padding

    if pandas:
        output = pd.DataFrame(counts.flatten(), columns=["counts"])
        output["stimulus"] = np.repeat(np.arange(number_of_stimuli), columns)
        output["response"] = np.tile(
            np.repeat(np.arange(n_responses), columns / number_of_stimuli),
            number_of_stimuli,
        )
        counts = output
    return counts
