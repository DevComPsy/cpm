import numpy as np
import pandas as pd
import warnings

from cpm.utils.metai import h2, acc_counts, H2r, conf_binned, bin_counts, H2r_given_c, dprime
from scipy.stats import norm
from scipy.integrate import quad


def metaI_d_prime(
        d_prime: float, 
        bins: int | None , 
        n_trials: int = 1_000_000, 
        seed: int = 0
):
    """
    Theoretical meta-I(d') of an ideal naive Bayesian observer given d'.
    Efficiency normalizer as proposed by Dayan (2023), later used to calculate meta_Ir1.

    Parameters
    ----------
    d_prime :
        Type 1 perceptual sensitivity, output of d_prime().
    bins :
        * None: computes metaI(d') for a naive Bayesian observer with continous (infinite) confidence
        through exact numerical integration.
        * int: confidence is quantile binned and Monte Carlo simulation is run for parameter estimation.
    
    n_trials:
        the number of trials of the Monte Carlo simulation ran if bins=int
        to approximate/generate the ideal confidence for a given d'. 
        Default is set to `1_000_000 `.

    seed: int
        default = 0

    Returns
    -------
    metaI_d_prime : float
    Theoretical meta-I(d') of an ideal naive Bayesian observer.

    Raises
    ------
    ValueError 
        if d_prime is not finite or <= 0.
    
    """
    if not np.isfinite (d_prime) or d_prime <= 0:
        raise ValueError ("d' needs to be > 0")

    sd = 2/d_prime
    
    # total expected conditional entropy for bins = None
    if bins is None:
        def integrand(x):
            prob_x = norm.pdf(x, loc = 1, scale = sd) + norm.pdf (-x, loc = 1, scale = sd)
            c = 1/ (1 + np.exp(-(d_prime)**2 * x /2))
            return prob_x * h2(c)
        h_rc, _ = quad(integrand, 0, 1 + 12 * sd) # upper limit covers 12 SDs
        return float(h2(norm.cdf(d_prime/2))) - h_rc
    
    # if bins = int 
    # true stimuli [-x,x] drawn randomly 
    rng = np.random.default_rng(seed)
    # assumes equal prior p(S1) = p(S2) = 0.5
    d = rng.choice([-1, +1], n_trials)

    # internal evidence α ~ N(d, (2/d′)²)
    alpha = rng.normal (loc=d, scale = sd)

    # primary choice a with accuracy r
    a = np.sign(alpha)
    accuracy = (a == d).astype(int)

    # ideal bayesian confidence c(alpha)
    confidence = 1/ (1 + np.exp(-(d_prime)**2 * np.abs(alpha)/2))
    
    # generate bincounts
    n_correct_prime, n_incorrect_prime, _ = bin_counts(accuracy, confidence, bins)
    
    return H2r(*acc_counts(accuracy)) - H2r_given_c(n_correct_prime, n_incorrect_prime)

def fit_metaI (
        accuracy: np.ndarray, 
        confidence: np.ndarray, 
        bins: int, 
        d_prime: float | None = None, 
        matched_bins: bool=False
):
    """
    Estimates metacognitive parameters using meta-I as proposed by Dayan (2023).

    Parameters
    ---------
    accuracy : list | np.ndarray
        0 for incorrect, 1 for correct.
    confidence : list | np.ndarray
        continous confidence rating per trial.
    bins : int
        number of bins used to bin the confidence ratings.
    d_prime : float
        type 1 perceptual sensitivity, output of function dprime(). Default is `None`
    matched_bins : bool
        *True: the same amount of bins as used in meta_I are applied, 
        advised if meta-I(d') shoulde be compared to meta_I (as in meta_Ir1).

    Returns
    ------
    results :
        *`'H2r'`: entropy of the accuracy,
        *`'H2r_given_c'`: entropy of the accuracy after seeing confidence,
        *`'meta_I'`: mutual information between accuracy and confidence in bits.
        *`'metaI_d_prime'`: meta-I(d') of an ideal naive Bayesian observer for a given d',
        *`'meta_Ir1'`: 1st measure of metacognitive efficiency (meta_I/metaI_dp),
        *`'meta_Ir2'`: 2nd measure of metacognitive efficiency (meta_I/h2_r),
        *`'n_correct'` : total number of correct trials,
        *`'n_incorrect'`: total number of incorrect trials,
        *`'n_correct_bin'`: number of correct trials per confidence bin
        *`'n_incorrect_bin'`: number of incorrect trials per confidence bin
        *`'thresholds'`: cuttings points used to create the bins,
        *`'rebin'`: 1 if `bin_ratings` had to be re-binned,
    
    """
    n_correct, n_incorrect = acc_counts(accuracy)
    h2_r = H2r(n_correct, n_incorrect)
    n_correct_bin, n_incorrect_bin, info = bin_counts(accuracy, confidence, bins)
    h2r_c = H2r_given_c(n_correct_bin, n_incorrect_bin)
    meta_I = float(h2_r - h2r_c)

    metaI_dp = (
        np.nan if d_prime is None
        else metaI_d_prime(d_prime, bins=bins if matched_bins else None)
    )
    return {
        "H2r": float(h2_r),
        "H2r_given_c": float(h2r_c),
        "meta_I": meta_I,
        "metaI_d_prime": metaI_dp,
        "meta_Ir1": meta_I/metaI_dp if np.isfinite(metaI_dp) and metaI_dp > 0 else np.nan,
        "meta_Ir2": (meta_I/h2_r) if h2_r > 0 else np.nan,
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "n_correct_bin": n_correct_bin.tolist(),
        "n_incorrect_bin": n_incorrect_bin.tolist(),
        "thresholds": info["thresholds"],
        "rebin": info["rebin"],
    }

class EstimatorMetaI:
    """
    Class to estimate the metacognitive parameters using meta-I as proposed by Dayan (2023).

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the data to be analyzed. See Note below.
    bins: int
        Number of bins to turn continous confidence rating into discrete bins.
    matched_bins : bool
        *True: the same amount of bins as used in meta_I are applied, 
        advised if meta-I(d') shoulde be compared to meta_I (as in meta_Ir1). 
    ppt_identifier: str
        column which denotes participants in the data
    
    Returns
    An EstimatorMetaI object.
    ------

    Notes
    -----
    The data DataFrame should contain following columns:
        - 'participant': Identifier for each participant.
        - 'signal' (integer): Stimulus that was presented to the participant, for example, 0 for S1 and 1 for S2.
        - 'response' (integer): Participant's response to the stimulus.
        - 'confidence' (integer, float): Participant's confidence rating for their response.
        - 'accuracy' (integer): Accuracy of the participant's response. 0 = incorrect, 1 = correct.
    
    """
    def __init__(
            self, 
            data, 
            bins,
            matched_bins =False,
            ppt_identifier="participant"
    ):
        self.data = data
        self.bins = bins
        self.matched_bins = matched_bins
        self.ppt_identifier = ppt_identifier
        self.fit = []

    def estimate(self):
        """
        Estimates metacognitive parameters for each particiupants using the meta-I model.

        Warns
        -----
        RunTimeWarning
            If a participant's d' <= 0, metaI_d_prime and meta_Ir1 will be Nan.
            The remaining measures for this participant (meta_I, meta_Ir2) are still computed.
        
        Raises
        -----
        RunTimeError
            If an unexpected error occurs during fitting on the data for a participant.
            
        """
        self.fit = []
        for ppt, group in self.data.groupby(self.ppt_identifier):
            dp = dprime(group["signal"].values, group["response"].values)

            if dp <= 0:
                warnings.warn(
                    f"Participant {ppt}: d'= {dp:.3f} <= 0, "
                    "metaI_d_prime and meta_I1r will be Nan.",
                    RuntimeWarning,
                    stacklevel = 2
                )
                dp = None

            try:
                result = fit_metaI(
                    accuracy=group["accuracy"].values,
                    confidence=group["confidence"].values,
                    bins=self.bins,
                    d_prime=dp,
                    matched_bins=self.matched_bins,
                )
            except Exception as e:
                raise RuntimeError (f"Error for participant {ppt}: {e}")
            result["ppt"] = ppt
            self.fit.append(result)

    def export(self):
        """
        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the metacognitive parameters.
        """
        return pd.DataFrame(self.fit)