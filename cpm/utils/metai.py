import numpy as np
import pandas as pd
import cpm
from cpm.utils.metad import bin_ratings
from scipy.stats import norm

def h2(p
):
    """
    Helper function to calculate the entropy of a random (Bernoulli) variable in bits.

    Parameters
    ----------
    p :
        probability of a random Bernoulli variable, must be [0, 1].
    
    Returns
    -------
    h2(p) : 
        the entropy of a random Bernoulli variable p in bits.
    
    Example
    -------
    >>> h2([0, 0.5, 1])   
    returns array ([0., 1., 0.])

    """
    p = np.asarray(p, dtype=float)
    q = 1 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -(np.where(p > 0, p * np.log2(p), 0.0)
                + np.where(q > 0, q * np.log2(q), 0.0))
    return out

def acc_counts(
        accuracy: np.ndarray
):
    """
    Helper function which counts the number of correct/incorrect trials.

    Parameters
    ----------
    accuracy : list | np.ndarray
        0 for incorrect, 1 for correct trials.

    Returns
    -------
    n_correct, n_incorrect : int

    """
    accuracy = np.asarray(accuracy).astype(int)
    return int((accuracy == 1).sum()), int((accuracy == 0).sum())

def H2r(
    n_correct: np.asarray, 
    n_incorrect: np.asarray
):
    """
    Calculates the entropy of accuracy.
    
    Parameters
    ----------
    n_correct, n_incorrect : int
    number of correct and incorrect trials from the acc_counts function.
    
    Returns
    -------
    H2r : the entropy of accuracy.

    """
    n_correct = np.asarray(n_correct, dtype=float)
    n_incorrect = np.asarray(n_incorrect, dtype=float)
    return float(h2(n_correct.sum() / (n_correct.sum() + n_incorrect.sum())))

def conf_binned(
    confidence: np.ndarray,
    bins: int,  
    ignore_invalid: bool = False 
):
    """
    Converts continous confidence ratings into discrete 1...n levels.
    Wrapper function of bin_ratings function in metad.py
    
    Parameters
    ----------
    confidence : list | np.ndarray
        Continous confidence rating per trial.
    bins : int
        Number of discrete bins. Must be an integer >= 2.
    ignore_invalid : bool
        If `False` (default), raises an error if the discretisation process is not possible.
        Passed directly on to `bin_ratings`.
 
    Returns
    -------
    levels : np.ndarray
        New rating array only containing integers between 1 and `nbins`.
    info : dict
        Dictionary containing logs of the binning process:
        *`'n_levels'` : int - number of bins used.
        *`'thresholds'` : list or None - cuttings points used to create the bins.
        *`'rebin'` : int - 1 if `bin_ratings` had to be re-binned due to ties at either low/high values.
    
    ..warning:: Automatically controls for bias in high or low confidence ratings, as inherited from `bin_ratings`.

    Raises
    ------
    ValueError
        If `bins` is not an int >= 2, or if confidence contains too many identical values,
        such that both the lowest and highest quantiles are equal and `ignore_invalid` is `False`.

    Notes
    -----
    Quantile bins used from `cpm.utils.metad.bin_ratings` from meta-d' estimator.
    
    """
    if isinstance(bins, bool) or not isinstance (bins, (int, np.integer)) or bins < 2:
          raise ValueError("`bins` must be an integer >=2.")
    confidence = np.asarray(confidence, dtype = float)
    levels, out = bin_ratings(
                ratings=confidence,
                nbins=int(bins),
                verbose=False,
                ignore_invalid=ignore_invalid,
            )
    edges = out.get("confBins")
    if edges is None:
         cuts = None
    else: 
         parts = edges if isinstance(edges, list) else [edges]
         cuts = np.concatenate ([np.atleast_1d(np.asarray(e, dtype=float)) for e in parts]).tolist()
    rebin = int(np.ravel(out.get("rebin", [0]))[0])
    return levels, {"n_levels": int(bins), "thresholds": cuts, "rebin": rebin}

def bin_counts(
    accuracy: np.ndarray, 
    confidence: np.ndarray, 
    bins: int
):
    """
    Counts the number of correct and incorrect trials per confidence bin.

    Parameters
    ----------
    accuracy : list | np.ndarray
        0 for incorrect, 1 for correct.
    confidence : list | np.ndarray
        continous confidence rating per trial.
    bins : int
        number of quantile bins, see `:func:`conf_binned.
        
    Returns
    -------
    bin_count_correct: np.ndarray
        array with the correct number of trials, number of entries for n number of confidence bins.
    bin_count_incorrect: np.ndarray
        array with the incorrect number of trials, number of entries for n number of confidence bins.
    info: dict
        Dictionary containing logs of the binning process:
        
    """
    levels, info = conf_binned(confidence, bins)
    n_levels = info["n_levels"]
    accuracy = np.asarray(accuracy).astype(int)
    idx = np.asarray(levels).astype(int) - 1
    bin_count_correct = np.bincount(idx[accuracy == 1], minlength = n_levels)
    bin_count_incorrect = np.bincount(idx[accuracy == 0], minlength = n_levels)

    return bin_count_correct, bin_count_incorrect, info

def H2r_given_c (
        bin_count_correct: np.ndarray, 
        bin_count_incorrect: np.ndarray
):
    """
    Parameters
    ---------
    bin_count_correct : np.ndarray
        number of correct trials in each confidence bin.
    bin_count_incorrect : np.ndarray
        number of incorrect trials in each confidence bin.

    Returns
    ------
    H2r_given_c : np.ndarray
        Entropy of the accuracy after seeing confidence H2(r|c).
        
    """
    n_bin = bin_count_correct + bin_count_incorrect
    p_cbin = n_bin / n_bin.sum()
    p_corr_given_c = np.where (n_bin > 0, bin_count_correct / n_bin(n_bin > 0, 1), 0.0) #in case of empty bins
    return np.sum(p_cbin * h2(p_corr_given_c))

def dprime(
    signal: np.ndarray, 
    response: np.ndarray, 
    correction: float = 0.5 
):
    """
    Determines d' (type 1 perceptual sensitivity in SDT) using scipys norm.ppf
    by z-transforming Hit and False Alarm rate. d'= z(Hit) - z(FA)

    Parameters
    ----------
    signal : list | np.ndarray
        stimulus S1 presented signal = 0, stimulus S2 presented signal = 1.
    response : list | np.ndarray
        participant choose S1 response = 0, participant choose S2 response = 1.
    correction : float
                *correction of `0.5` (loglinear) 
                eqivalent of metad.py count_trials default padding padAmount= 1/(2*nRatings), 
                which adds +0.5 to hit/FA counts and +1 to the total trial counts.

                *correction of `1` (same as in meta-d)
                equivalent to EstimatorMetaD correction padAmount = 1/nRatings, 
                which adds +1 to hit/FA counts and +2 to the total trial counts.

                Default is set to `0.5`.

    Returns
    -------
    d_prime: float
        Type 1 perceptual sensitivity.

    Notes
    -----
    To prevent d' -> ± ∞ for hits or fa of 0 or 1 respectivley a correction equivalent to padAmount is used. 
    If metacognitive efficiency type 1 measure should be directly comparable to the M-ratio: choose correction of 1. 
    If meta-I(d')/metacognitive efficiency type 1 is analyzed on it's own: correction of 0.5 produces the least distortions of d'.

    """
    hits = (response [signal == 1] == 1).sum()
    fa = (response [signal == 0] == 1).sum()
    hit_rate = (hits + correction)/((signal == 1).sum() + 2 * correction)
    fa_rate = (fa + correction)/((signal == 0).sum() + 2 * correction)
    z_hit, z_fa = norm.ppf([hit_rate, fa_rate])
    return z_hit - z_fa