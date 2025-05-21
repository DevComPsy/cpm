# The majority of this code was adapted from the metadpy package
# and the original metad code from: http://www.columbia.edu/~bsm2105/type2sdt
# Information about the license can be found in the LICENSE-BUNDLE file

import warnings
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
import cpm
from scipy.optimize import SR1, Bounds, LinearConstraint, minimize
from scipy.stats import norm, multivariate_normal

from cpm.core.optimisers import numerical_hessian, prepare_data
from cpm.core.data import detailed_pandas_compiler, decompose
from cpm.core.parallel import detect_cores, execute_parallel
from cpm.utils.metad import count_trials, bin_ratings


def metad_nll(
    guess: np.ndarray,
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    nRatings: int,
    d1: float,
    t1c1: float,
    s: int,
    parameters: cpm.generators.Parameters = None,
    prior: bool = False,
):
    """
    Evaluate the negative log-likelihood of the parameters given the data.
    This function is used in the optimization process to fit the meta-d model.

    Parameters
    ----------
    guess :
        guess[0] = meta d'
        guess[1:end] = type-2 criteria locations
    nR_S1, nR_S2 :
        These are vectors containing the total number of responses in each response
        category, conditional on presentation of S1 and S2.
    nRatings :
        Numbers of ratings.
    d1 :
        d prime.
    t1c1 :
        The type-1 criterion.
    s :
        Ratio of standard deviations for type 1 distributions as:
        `s = np.std(S1) / np.std(S2)`. If not specified, s is set to a default
        value of 1. For most purposes, it is recommended to set `s=1`. See
        http://www.columbia.edu/~bsm2105/type2sdt for further discussion.

    Returns
    -------
    logL : float
        The negative log-likelihood of the parameters given the data.

    """
    meta_d1, t2c1 = guess[0], guess[1:]

    # define mean and SD of S1 and S2 distributions
    S1mu = -meta_d1 / 2
    S1sd = 1
    S2mu = meta_d1 / 2
    S2sd = S1sd / s

    # adjust so that the type 1 criterion is set at 0
    # (this is just to work with optimization toolbox constraints...
    #  to simplify defining the upper and lower bounds of type 2 criteria)
    S1mu = S1mu - (meta_d1 * (t1c1 / d1))
    S2mu = S2mu - (meta_d1 * (t1c1 / d1))

    t1c1 = 0

    # set up MLE analysis
    # get type 2 response counts
    # S1 responses
    nC_rS1, nI_rS1 = [], []
    for i in range(nRatings):
        nC_rS1.append(nR_S1[i])
        nI_rS1.append(nR_S2[i])

    # S2 responses
    nC_rS2 = [nR_S2[i + nRatings] for i in np.arange(nRatings)]
    nI_rS2 = [nR_S1[i + nRatings] for i in np.arange(nRatings)]

    # get type 2 probabilities
    C_area_rS1 = norm.cdf(t1c1, S1mu, S1sd)
    I_area_rS1 = norm.cdf(t1c1, S2mu, S2sd)

    C_area_rS2 = 1 - norm.cdf(t1c1, S2mu, S2sd)
    I_area_rS2 = 1 - norm.cdf(t1c1, S1mu, S1sd)

    t2c1x = [-np.inf]
    t2c1x.extend(t2c1[0 : (nRatings - 1)])
    t2c1x.append(t1c1)
    t2c1x.extend(t2c1[(nRatings - 1) :])
    t2c1x.append(np.inf)

    prC_rS1 = [
        (norm.cdf(t2c1x[i + 1], S1mu, S1sd) - norm.cdf(t2c1x[i], S1mu, S1sd))
        / C_area_rS1
        for i in range(nRatings)
    ]
    prI_rS1 = [
        (norm.cdf(t2c1x[i + 1], S2mu, S2sd) - norm.cdf(t2c1x[i], S2mu, S2sd))
        / I_area_rS1
        for i in range(nRatings)
    ]

    prC_rS2 = [
        (
            (1 - norm.cdf(t2c1x[nRatings + i], S2mu, S2sd))
            - (1 - norm.cdf(t2c1x[nRatings + i + 1], S2mu, S2sd))
        )
        / C_area_rS2
        for i in range(nRatings)
    ]
    prI_rS2 = [
        (
            (1 - norm.cdf(t2c1x[nRatings + i], S1mu, S1sd))
            - (1 - norm.cdf(t2c1x[nRatings + i + 1], S1mu, S1sd))
        )
        / I_area_rS2
        for i in range(nRatings)
    ]

    # calculate logL
    logL = 0.0
    for i in range(nRatings):
        logL = (
            logL
            + nC_rS1[i] * np.log(prC_rS1[i])
            + nI_rS1[i] * np.log(prI_rS1[i])
            + nC_rS2[i] * np.log(prC_rS2[i])
            + nI_rS2[i] * np.log(prI_rS2[i])
        )

    if prior:
        parameters.update(**{
            "meta_d": meta_d1,
            "criterion_type2": t2c1,
            "criterion_type1": t1c1,
            "s": s,
            "d_prime": d1,
        })
        logL += parameters.PDF(log=True)

    # returning -inf may cause optimize.minimize() to fail
    if np.isinf(logL) or np.isnan(logL):
        logL = -1e300

    return -logL

def fit_metad(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    nRatings: int,
    nCriteria: Optional[int] = None,
    s: int = 1,
    verbose: int = 0,
    fninv: Callable = norm.ppf,
    fncdf: Callable = norm.cdf,
    parameters: cpm.generators.Parameters = None,
    prior: bool = False,
):
    """
    Estimate metacognitive parameters using the meta-d model proposed by Maniscalco and Lau (2012).
    This function fits the meta-d model to the data and returns the estimated parameters.
    It is done via maximum likelihood estimation (MLE) using the scipy.optimize.minimize function.
    The function uses the trust-constr method for optimization, which is suitable for constrained
    optimization problems.

    Parameters
    ----------
    nR_S1, nR_S2 :
        These are vectors containing the total number of responses in
        each response category, conditional on presentation of S1 and S2. If
        nR_S1 = [100, 50, 20, 10, 5, 1], then when stimulus S1 was presented, the
        subject had the following response counts:
        * responded `'S1'`, rating=`3` : 100 times
        * responded `'S1'`, rating=`2` : 50 times
        * responded `'S1'`, rating=`1` : 20 times
        * responded `'S2'`, rating=`1` : 10 times
        * responded `'S2'`, rating=`2` : 5 times
        * responded `'S2'`, rating=`3` : 1 time

        The ordering of response / rating counts for S2 should be the same as
        it is for S1. e.g. if nR_S2 = [3, 7, 8, 12, 27, 89], then when stimulus S2
        was presented, the subject had the following response counts:
        * responded `'S1'`, rating=`3` : 3 times
        * responded `'S1'`, rating=`2` : 7 times
        * responded `'S1'`, rating=`1` : 8 times
        * responded `'S2'`, rating=`1` : 12 times
        * responded `'S2'`, rating=`2` : 27 times
        * responded `'S2'`, rating=`3` : 89 times
    nRatings :
        Number of discrete ratings. If a continuous rating scale was used, and
        the number of unique ratings does not match `nRatings`, will convert to
        discrete ratings using :py:func:`metadpy.utils.bin_ratings`.
        Default is set to 4.
    nCriteria :
        (Optional) Number criteria to be fitted. If `None`, the number of criteria is
        set to `nCriteria = int(2 * nRatings - 1)`.
    s :
        Ratio of standard deviations for type 1 distributions as:
        `s = np.std(S1) / np.std(S2)`. If not specified, s is set to a default
        value of 1. For most purposes, it is recommended to set `s=1`. See
        http://www.columbia.edu/~bsm2105/type2sdt for further discussion.
    verbose :
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
            * 3 : display progress during iterations (more complete report).
    fninv :
        A function handle for the inverse CDF of the type 1 distribution. If
        not specified, fninv defaults to :py:func:`scipy.stats.norm.ppf()`.
    fncdf :
        A function handle for the CDF of the type 1 distribution. If not
        specified, fncdf defaults to :py:func:`scipy.stats.norm.cdf()`.

    Returns
    -------
    results :
        In the following, S1 and S2 represent the distributions of evidence generated
        by stimulus classes S1 and S2:
        * `'d'` : d-prime, the distance between the means of the S1 and S2 distributions, in RMS units.
        * `'s'` : ratio of the standard deviations of the S1 and S2
        * `'meta_d'` : meta-d' in RMS units
        * `'m_diff'` : `meta_d - d`
        * `'m_ratio'` : `meta_d / d`
        * `'meta_c'` : type 1 criterion for meta-d' fit, RMS units.
        * `'t2ca_rS1'` : type 2 criteria of "S1" responses for meta-d' fit, RMS units.
        * `'t2ca_rS2'` : type 2 criteria of "S2" responses for meta-d' fit, RMS units.
        * `'logL'` : log likelihood of the data fit
        * `'est_HR2_rS1'` : estimated (from meta-d' fit) type 2 hit rates for S1 responses.
        * `'obs_HR2_rS1'` : actual type 2 hit rates for S1 responses.
        * `'est_FAR2_rS1'` : estimated type 2 false alarm rates for S1 responses.
        * `'obs_FAR2_rS1'` : actual type 2 false alarm rates for S1 responses.
        * `'est_HR2_rS2'` : estimated type 2 hit rates for S2 responses.
        * `'obs_HR2_rS2'` : actual type 2 hit rates for S2 responses.
        * `'est_FAR2_rS2'` : estimated type 2 false alarm rates for S2 responses.
        * `'obs_FAR2_rS2'` : actual type 2 false alarm rates for S2 responses.

    """
    if nCriteria is None:
        nCriteria = int(2 * nRatings - 1)

    # parameters
    # meta-d' - 1
    # t2c - nCriteria-1
    # constrain type 2 criteria values, such that t2c(i) is always <= t2c(i+1)
    # -->  t2c(i+1) >= t2c(i) + 1e-5 (i.e. very small deviation from equality)
    A, ub, lb = [], [], []
    for ii in range(nCriteria - 2):
        tempArow: List[int] = []
        tempArow.extend(np.zeros(ii + 1))
        tempArow.extend([1, -1])
        tempArow.extend(np.zeros((nCriteria - 2) - ii - 1))
        A.append(tempArow)
        ub.append(-1e-5)
        lb.append(-np.inf)

    # lower bounds on parameters
    LB = []
    LB.append(-10.0)  # meta-d'
    LB.extend(-20 * np.ones((nCriteria - 1) // 2))  # criteria lower than t1c
    LB.extend(np.zeros((nCriteria - 1) // 2))  # criteria higher than t1c

    # upper bounds on parameters
    UB = []
    UB.append(10.0)  # meta-d'
    UB.extend(np.zeros((nCriteria - 1) // 2))  # criteria lower than t1c
    UB.extend(20 * np.ones((nCriteria - 1) // 2))  # criteria higher than t1c

    # set up initial guess at parameter values
    ratingHR = []
    ratingFAR = []
    for c in range(1, int(nRatings * 2)):
        ratingHR.append(sum(nR_S2[c:]) / sum(nR_S2))
        ratingFAR.append(sum(nR_S1[c:]) / sum(nR_S1))

    # obtain index in the criteria array to mark Type I and Type II criteria
    t1_index = nRatings - 1
    t2_index = list(set(list(range(0, 2 * nRatings - 1))) - set([t1_index]))

    d1 = (1 / s) * fninv(ratingHR[t1_index]) - fninv(ratingFAR[t1_index])
    meta_d1 = d1

    c1 = (-1 / (1 + s)) * (fninv(ratingHR) + fninv(ratingFAR))
    t1c1 = c1[t1_index]
    t2c1 = c1[t2_index]

    # initial values for the minimization function
    guess = [meta_d1]
    guess.extend(list(t2c1 - (meta_d1 * (t1c1 / d1))))

    # other inputs for the minimization function
    bounds = Bounds(LB, UB)
    linear_constraint = LinearConstraint(A, lb, ub)

    # minimization of negative log-likelihood
    results = minimize(
        metad_nll,
        guess,
        args=(nR_S1, nR_S2, nRatings, d1, t1c1, s, parameters, prior),
        method="trust-constr",
        jac="2-point",
        hess=SR1(),
        constraints=[linear_constraint],
        options={"verbose": verbose},
        bounds=bounds,
    )
    # quickly process some of the output
    meta_d1 = results.x[0]
    t2c1 = results.x[1:] + (meta_d1 * (t1c1 / d1))
    logL = -results.fun

    # I_nR and C_nR are rating trial counts for incorrect and correct trials
    # element i corresponds to # (in)correct w/ rating i
    I_nR_rS2 = nR_S1[nRatings:]
    I_nR_rS1 = list(np.flip(nR_S2[0:nRatings], axis=0))

    C_nR_rS2 = nR_S2[nRatings:]
    C_nR_rS1 = list(np.flip(nR_S1[0:nRatings], axis=0))

    obs_FAR2_rS2 = [
        sum(I_nR_rS2[(i + 1) :]) / sum(I_nR_rS2) for i in range(nRatings - 1)
    ]
    obs_HR2_rS2 = [
        sum(C_nR_rS2[(i + 1) :]) / sum(C_nR_rS2) for i in range(nRatings - 1)
    ]
    obs_FAR2_rS1 = [
        sum(I_nR_rS1[(i + 1) :]) / sum(I_nR_rS1) for i in range(nRatings - 1)
    ]
    obs_HR2_rS1 = [
        sum(C_nR_rS1[(i + 1) :]) / sum(C_nR_rS1) for i in range(nRatings - 1)
    ]

    # find estimated t2FAR and t2HR
    S1mu = -meta_d1 / 2
    S1sd = 1
    S2mu = meta_d1 / 2
    S2sd = S1sd / s

    mt1c1 = meta_d1 * (t1c1 / d1)

    C_area_rS2 = 1 - fncdf(mt1c1, S2mu, S2sd)
    I_area_rS2 = 1 - fncdf(mt1c1, S1mu, S1sd)

    C_area_rS1 = fncdf(mt1c1, S1mu, S1sd)
    I_area_rS1 = fncdf(mt1c1, S2mu, S2sd)

    est_FAR2_rS2, est_HR2_rS2 = [], []
    est_FAR2_rS1, est_HR2_rS1 = [], []

    for i in range(nRatings - 1):

        t2c1_lower = t2c1[(nRatings - 1) - (i + 1)]
        t2c1_upper = t2c1[(nRatings - 1) + i]

        I_FAR_area_rS2 = 1 - fncdf(t2c1_upper, S1mu, S1sd)
        C_HR_area_rS2 = 1 - fncdf(t2c1_upper, S2mu, S2sd)

        I_FAR_area_rS1 = fncdf(t2c1_lower, S2mu, S2sd)
        C_HR_area_rS1 = fncdf(t2c1_lower, S1mu, S1sd)

        est_FAR2_rS2.append(I_FAR_area_rS2 / I_area_rS2)
        est_HR2_rS2.append(C_HR_area_rS2 / C_area_rS2)

        est_FAR2_rS1.append(I_FAR_area_rS1 / I_area_rS1)
        est_HR2_rS1.append(C_HR_area_rS1 / C_area_rS1)

    # package output
    fit = {}

    # fit["dprime"] = np.sqrt(2 / (1 + s**2)) * s * d1
    # fit["s"] = s
    # fit["meta_d"] = np.sqrt(2 / (1 + s**2)) * s * meta_d1
    # fit["meta_ca"] = (np.sqrt(2) * s / np.sqrt(1 + s**2)) * mt1c1
    # t2ca = (np.sqrt(2) * s / np.sqrt(1 + s**2)) * np.array(t2c1)
    # fit["t2ca_rS1"] = t2ca[0 : nRatings - 1]
    # fit["t2ca_rS2"] = t2ca[(nRatings - 1) :]

    fit["x"] = results.x
    fit["d"] = d1
    fit["meta_d"] = meta_d1
    fit["m_diff"] = fit["meta_d"] - fit["d"]
    fit["m_ratio"] = fit["meta_d"] / fit["d"]
    fit["s"] = s
    fit["meta_c"] = mt1c1
    fit["t2c1_rS1"] = t2c1[0 : nRatings - 1]
    fit["t2c1_rS2"] = t2c1[(nRatings - 1) :]
    fit["logL"] = logL

    fit["est_HR2_rS1"] = est_HR2_rS1
    fit["obs_HR2_rS1"] = obs_HR2_rS1

    fit["est_FAR2_rS1"] = est_FAR2_rS1
    fit["obs_FAR2_rS1"] = obs_FAR2_rS1

    fit["est_HR2_rS2"] = est_HR2_rS2
    fit["obs_HR2_rS2"] = obs_HR2_rS2

    fit["est_FAR2_rS2"] = est_FAR2_rS2
    fit["obs_FAR2_rS2"] = obs_FAR2_rS2

    return fit


class EstimatorMetaD:
    """
    Class to estimate metacognitive parameters using the meta-d model proposed by Maniscalco and Lau (2012).

    Parameters
    ----------
    data : pd.DataFrame, pd.DataFrameGroupBy
        DataFrame containing the data to be analyzed. See Note below.
    bins : int
        Number of bins to use for binning the confidence ratings.
    cl : int, optional
        Number of cores to use for parallel processing. If None, all available cores will be used.
    parallel : bool, default False
        If True, parallel processing will be used.
    libraries : list of str, default ["numpy", "pandas"]
        List of libraries to use for parallel processing in Jupyter. Default is ["numpy", "pandas"].
    prior : bool, default False
        If True, the log likelihoods will incorporate prior density of parameters.
    display : int, default 0
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
            * 3 : display progress during iterations (more complete report).
    ppt_identifier : str, optional
        Identifier for participants in the data. If None, the default identifier will be used.
    ignore_invalid : bool, default False
        If True, invalid confidence ratings will be ignored during binning. If False, an error will be raised if invalid ratings are found. We recommend setting this to False (Default).
    **kwargs : additional keyword arguments
        Additional keyword arguments to be passed to the optimization function.

    Returns
    -------
    An EstimatorMetaD object.

    Note
    ----
    The data DataFrame should contain the following columns:
        - 'participant': Identifier for each participant.
        - 'signal' (integer): Stimulus presented to the participant, for example, 0 for S1 and 1 for S2.
        - 'response' (integer): Participant's response to the stimulus.
        - 'confidence' (integer, float): Participant's confidence rating for their response.
        - 'accuracy' (integer): Accuracy of the participant's response. 0 = incorrect, 1 = correct.
    """
    def __init__(
        self,
        data=None,
        bins=None,
        cl=None,
        parallel=False,
        libraries=["numpy", "pandas"],
        prior=False,
        display=False,
        ppt_identifier=None,
        ignore_invalid=False,
        **kwargs,
    ):
        self.data = data
        self.bins = bins
        self.ppt_identifier = ppt_identifier
        self.data, self.participants, self.groups, self.__pandas__ = prepare_data(
            data, self.ppt_identifier
        )

        self.prior = prior
        self.kwargs = kwargs
        self.display = display
        self.prior = prior
        self.ignore_invalid = ignore_invalid
        self.fit = []
        self.details = []
        self.criteria = (bins - 1) * 2
        self.parameters = cpm.generators.Parameters(
            meta_d=cpm.generators.Value(
                value=np.random.randn(),
                lower=-10,
                upper=10,
                prior="norm",
                args={"mean": 1, "sd": 2},
            ),
            criterion_type2=cpm.generators.Value(
                value=np.random.randn(2 * bins - 2),
                lower=np.array([[-20] * (bins - 1), [0] * (bins - 1)]).flatten(),
                upper=np.array([[0] * (bins - 1), [20] * (bins - 1)]).flatten(),
                prior=multivariate_normal,
                args={
                    "mean": np.delete(np.linspace(-4, 4, 2 * bins - 1), bins - 1),
                    "cov": 3 * np.eye(2 * bins - 2),
                },
            ),
            s=1,
            criterion_type1=cpm.generators.Value(
                value=np.random.randn(),
                lower=-10,
                upper=10,
                prior="norm",
                args={"mean": 1, "sd": 2},
            ),
            d_prime=cpm.generators.Value(
                value=np.random.randn(),
                lower=-10,
                upper=10,
                prior="norm",
                args={"mean": 1, "sd": 2},
            ),
            bins=bins,
        )
        
        self.__parallel__ = parallel
        self.__libraries__ = libraries

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = detect_cores()

    def optimise(self):
        """
        Estimates the metacognitive parameters using the meta-d model.
        Here, we use a Trust-Region Constrained Optimization algorithm (Conn et al., 2000) to fit the model to the data.
        We use the `trust-constr` method from `scipy.optimize.minimize` to perform the optimization, and minimise the negative log-likelihood of the data given the model parameters.
        The optimization is performed for each participant in the data.

        Notes
        -----
        If you want to tune the behaviour of the optimization, you can do so by passing additional keyword arguments to the class constructor. See the [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html) documentation for more details on the available options. By default, the optimization will use the `trust-constr` method with the default options specified in the `scipy.optimize.minimize` documentation.


        References
        ----------
        Conn, A. R., Gould, N. I. M., & Toint, P. L. (2000). Trust Region Methods. Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898719857

        """

        def __task(participant, **args):
            subject, _, ppt = decompose(
                participant=participant,
                pandas=self.__pandas__,
                identifier=self.ppt_identifier,
            )

            try:
                subject["discrete"] = bin_ratings(
                    ratings=subject.confidence.to_numpy(),
                    nbins=self.bins,
                    ignore_invalid=self.ignore_invalid)[0]
            except Exception as e:
                raise RuntimeError(f"Error binning ratings for participant {ppt}: {e}")
            nRatings = self.bins
            nCriteria = int(2 * nRatings - 1)  # number criteria to be fitted
            
            nR_S1, nR_S2 = count_trials(
                data=subject,
                stimuli="signal",
                responses="response",
                accuracy="accuracy",
                confidence="discrete",
                nRatings=nRatings,
                padding=True,
                padAmount=1/nRatings,
            )
            
            results_dict = fit_metad(
                nR_S1=nR_S1,
                nR_S2=nR_S2,
                nRatings=nRatings,
                nCriteria=nCriteria,
                s=1,
                verbose=self.display,
                fninv=norm.ppf,
                fncdf=norm.cdf,
                parameters=self.parameters,
                prior=self.prior,
            )

            def f(x):
                return metad_nll(
                    guess=x,
                    nR_S1=nR_S1,
                    nR_S2=nR_S2,
                    nRatings=nRatings,
                    d1=results_dict["d"],
                    t1c1=results_dict["meta_c"],
                    s=1,
                    parameters=self.parameters,
                    prior=self.prior,
                )

            hess = numerical_hessian(
                func=f,
                params=results_dict["x"] + 1e-3,
            )

            results_dict.update({
                "hess": hess,
            })
            results_dict.update({
                "ppt": ppt,
            })
            ## remove x
            results_dict.pop("x")
            return results_dict

        if self.__parallel__:
            self.fit = execute_parallel(
                job=__task,
                data=self.data,
                cl=self.cl,
                method=None,
                pandas=self.__pandas__,
                libraries=self.__libraries__,
                **self.kwargs,
            )
        else:
            results = list(map(__task, self.data))
            self.fit = results
        return None

    def export(self):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters.
        """
        output = detailed_pandas_compiler(self.fit)
        output.reset_index(drop=True, inplace=True)
        return output
