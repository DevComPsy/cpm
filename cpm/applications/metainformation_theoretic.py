import numpy as np
import pandas as pd
import warnings

from scipy.stats import norm
from scipy.integrate import quad

from cpm.utils.metai import h2, acc_counts, H2r, conf_binned, bin_counts, H2r_given_c, dprime
from cpm.core.parallel import detect_cores
from joblib import Parallel, delayed

def _is_column(data, col):
    return isinstance(col, str) and col in data.columns

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
        signal: np.ndarray, 
        confidence: np.ndarray, 
        bins: int,
        response: np.ndarray | None = None,
        accuracy: np.ndarray | None = None, 
        d_prime: float | None = None, 
        matched_bins: bool=False,
        ignore_invalid: bool=False
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
        *`'d_prime`: d'
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
    
    Raises
    ------
    ValueError
        If neither `response` nor `accuracy` are provided; if the input `signal`, `response`, `accuracy` and `confidence` are of different lengths;
        if `response` and `accuracy` are inconsistent on a valid trial; if there is no valid trial at all.

    Notes
    -----
    Trials where `signal`, `response` or `accuracy` are not 0 or 1, or confidence is not finite,
    are automatically omitted.

    If only `accuracy` is provided, responses are inferred from it.
    If only `response` is provided, accuracy is inferred from it.
    If both `response` and `accuracy` are provided, they are checked for consistency, and a
    `ValueError` is raised on any trial where they disagree.

    """
    if response is None and accuracy is None:
        raise ValueError(
            "fit_metaI needs `response`, `accuracy`, or both; neither was provided."
        )

    signal = np.asarray(signal, dtype=float)
    confidence = np.asarray(confidence, dtype=float)

    inputs = [signal, confidence]

    if response is not None:
        response = np.asarray(response, dtype=float)
        inputs.append(response)
    if accuracy is not None:
        accuracy = np.asarray(accuracy, dtype=float)
        inputs.append(accuracy)
    if len({len(x) for x in inputs}) > 1:
        raise ValueError(
            "`stimuli`, `response`, `accuracy` and `confidence` must have the same "
            f"length; got lengths {[len(x) for x in inputs]}."
        )
    
    bin_signal = np.isin(signal, (0,1))
    out_of_range = ~bin_signal & ~np.isnan(signal)
    if out_of_range.any():
        warnings.warn(
            f"{out_of_range.sum()} trial(s) have signal values outside {{0, 1}} (values "
            f"{np.unique(signal[out_of_range])[:5].tolist()}) and are omitted; "
            "code signal as 0 for S1 and 1 for S2.",
            UserWarning,
            stacklevel=2,
        )
    
    _response_provided = response is not None
    _accuracy_provided = accuracy is not None
    
    bin_response = np.isin(response, (0,1)) if response is not None else None
    bin_accuracy = np.isin(accuracy, (0,1)) if accuracy is not None else None

    if accuracy is None:
        accuracy = np.where(
            bin_signal & bin_response,
            (signal == response).astype(int),
            np.nan,
        )

    if response is None:
        response = np.where(
            bin_signal & bin_accuracy,
            np.where(accuracy == 1, signal, 1 - signal),
            np.nan,
        )

    valid = (
        bin_signal
        & np.isin(response, (0, 1))
        & np.isin(accuracy, (0, 1))
        & np.isfinite (confidence)
    )

    if _response_provided and _accuracy_provided:
        mismatch = valid & ((response == signal) != (accuracy == 1))
        if mismatch.any():
            positions = np.flatnonzero(mismatch)
            raise ValueError(
                f"`response` and `accuracy` are inconsistent on {len(positions)} "
                f"trial(s), at positions {positions[:5].tolist()}"
                f"{' and more' if len(positions) > 5 else ''}: a response is "
                "correct when it equals the stimulus."
                        )
    if not valid.any():
        raise ValueError(
            "fit_metaI found no valid trials: `signal`, `response` and "
            "`accuracy` (if given) must be 0 or 1, and `confidence` must be finite. "
        )
    
    signal = signal[valid]
    response = response[valid]
    accuracy = accuracy[valid]
    confidence = confidence[valid]


    n_correct, n_incorrect = acc_counts(accuracy)
    h2_r = H2r(n_correct, n_incorrect)
    n_correct_bin, n_incorrect_bin, info = bin_counts(accuracy, confidence, bins, ignore_invalid=ignore_invalid)
    h2r_c = H2r_given_c(n_correct_bin, n_incorrect_bin)
    meta_I = float(h2_r - h2r_c)

    metaI_dp = (
        np.nan if d_prime is None
        else metaI_d_prime(d_prime, bins=bins if matched_bins else None)
    )
    return {
        "d_prime": d_prime,
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
    dprime_correction: float
            *correction of `0.5` (loglinear) 
            eqivalent of metad.py count_trials default padding padAmount= 1/(2*nRatings), 
            which adds +0.5 to hit/FA counts and +1 to the total trial counts.

            *correction of `1` (same as in meta-d)
            equivalent to EstimatorMetaD correction padAmount = 1/nRatings, 
            which adds +1 to hit/FA counts and +2 to the total trial counts.

            Default is set to `0.5`.
    cl : int, optional
            Number of cores to use for parallel processing. If None, all available cores will be used.
    parallel : bool, default False
            If True, parallel processing will be used.
    
    Returns
    ------
    An EstimatorMetaI object.

    Raises
    ------
    ValueError
        If `data` is not a DataFrame; if `signal`, `confidence` or a ppt_identifier are not provided
        as columns in the DataFrame. 
    
    ValueError
        If DataFrame doesn't contain either `response` or `accuracy`.
    
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
            ppt_identifier="participant",
            dprime_correction=0.5,
            cl=None,
            parallel=False,
    ):
        self.data = data
        self.bins = bins
        self.matched_bins = matched_bins
        self.ppt_identifier = ppt_identifier
        self.dprime_correction = dprime_correction
        self.fit = []

        self.__pandas__ = True
        self.__parallel__ = parallel

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = detect_cores()

        if isinstance(data, pd.DataFrame):
            for col in ["signal", "confidence", ppt_identifier]:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in DataFrame")
                
            self.signal = data["signal"].values
            self.confidence = data["confidence"].values
            self.response = data["response"].values if _is_column(data, "response") else None
            self.accuracy = data["accuracy"].values if _is_column(data, "accuracy") else None

            if self.response is None and self.accuracy is None:
                raise ValueError(
                    "DataFrame must contain either `response` or `accuracy`."
                )

        else:
            raise ValueError("`data` should be a DataFrame.")
            
    def estimate(self):
        """
        Estimates metacognitive parameters for each particiupants using the meta-I model.

        Warns
        -----
        RuntimeWarning
            If a participant's d' <= 0, metaI_d_prime and meta_Ir1 will be Nan.
            The remaining measures for this participant (meta_I, meta_Ir2) are still computed.
        
        UserWarning
            If `response` was not provided, d' computed from responses inferred from `accuracy`.

        RuntimeWarning
            If an unexpected error occurs during fitting, all metrics will be set to Nan for that participant.

        """
        self.fit = []

        if self.response is None:
            warnings.warn(
                "`response` was not provided, d' computed from responses inferred from `accuracy` for all participants.",
                UserWarning,
                stacklevel = 2,
            )

        def __task(ppt_group, **args):
            ppt, subject = ppt_group

            if self.response is not None:
                dp = dprime(subject["signal"].values, subject["response"].values, correction=self.dprime_correction)
            else:
                # infer responses from accuracy if missing, for d'
                sig = subject["signal"].values.astype(float)
                acc = subject["accuracy"].values.astype(float)
                inf_response = np.where(acc == 1, sig, 1 - sig)
                dp = dprime(subject["signal"].values, inf_response, correction = self.dprime_correction)

            dp_original = dp
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
                    signal=subject["signal"].values,
                    confidence=subject["confidence"].values,
                    accuracy=subject["accuracy"].values if self.accuracy is not None else None,
                    response=subject["response"].values if self.response is not None else None,
                    bins=self.bins,
                    d_prime=dp,
                    matched_bins=self.matched_bins,
                )
            except Exception as e:
                warnings.warn(
                    f"Participant {ppt}: unexpected error during fitting: {e}. All metrics will be Nan.",
                    RuntimeWarning,
                    stacklevel = 2
                )
                result = {
                "d_prime": np.nan,
                "H2r": np.nan,
                "H2r_given_c": np.nan,
                "meta_I": np.nan,
                "metaI_d_prime": np.nan,
                "meta_Ir1": np.nan,
                "meta_Ir2": np.nan,
                "n_correct": None,
                "n_incorrect": None,
                "n_correct_bin": np.nan,
                "n_incorrect_bin":np.nan,
                "thresholds": None,
                "rebin": None,
                }

            result["ppt"] = ppt
            result["d_prime"] = dp_original
            return result

        if self.__parallel__:
            self.fit = Parallel(n_jobs=self.cl)(
            delayed(__task)(ppt_group)
            for ppt_group in self.data.groupby(self.ppt_identifier)
            )

        else:
            self.fit = list(map(__task, self.data.groupby(self.ppt_identifier)))

    def export(self):
        """
        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the metacognitive parameters.
        """
        return pd.DataFrame(self.fit)