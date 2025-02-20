import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from cpm.utils.optimisation import LinearConstraint
from cpm.generators import Parameters, Value
from cpm.core.data import determine_data_length

__all__ = ["MetaHelper", "InitialiseParametersMeta", "MetaD"]


class MetaHelper:
    """
    Helper class for models calculating meta-d based on Signal Detection Theory. It computes the necessary parameters for the MetaModel class.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the data. It must contain columns for stimulus, response, confidence, and ppt.
    nbins : int
        Number of confidence bins.
    s : float
        Signal detection criterion.
    fncdf : function
        Cumulative distribution function. Default is scipy.stats.norm.cdf.
    fninv : function
        Inverse cumulative distribution function. Defasult is scipy.stats.norm.ppf.
    apply_adjustment : bool
        Apply adjustment to the counts.
    adjustment_val : float
        Adjustment value.

    """

    def __init__(
        self,
        data=None,
        nbins=4,
        s=1,
        fncdf=norm.cdf,
        fninv=norm.ppf,
        apply_adjustment=False,
        adjustment_val=None,
        ppt_identifier="ppt",
        **kwargs,
    ):
        self.data = data
        self.nbins = nbins
        self.s = s
        self.fncdf = fncdf
        self.fninv = fninv
        self.apply_adjustment = apply_adjustment
        self.adjustment_val = (
            adjustment_val if adjustment_val is not None else 1 / (2 * self.nbins)
        )
        self.ppt_identifier = ppt_identifier
        self.ppts = (
            self.data[self.ppt_identifier].unique()
            if self.ppt_identifier is not None
            else [0]
        )
        if data is not None:
            self.nR_S1, self.nR_S2 = self.bin_confidence()
            self.ratings_hit_rates, self.ratings_false_alarms = self.rates()
            self.d_prime = self.compute_d_prime()
            self.c1 = self.compute_c1()
            self.criterion_type1 = self.c1[:, self.nbins - 1, np.newaxis]
            self.criterion_type2 = self.c1 - self.criterion_type1
            self.criterion_type2 = np.delete(
                self.criterion_type2, self.nbins - 1, axis=1
            )
            self.data_pandas = self.create_df()

        self.kwargs = kwargs

    def new_bin_ratings(self, confidence_continuous):
        """
        Bins the confidence_continuous array into self.nbins based on quantiles, ensuring proper assignment of binned confidence levels.

        Parameters:
            confidence_continuous (numpy.ndarray): Array of sliding confidence values.

        Returns:
            confidence_discrete (numpy.ndarray): Array with binned confidence ratings.
            out (dict): Contains information about the confidence bins and rebinning status.
        """
        out = {}
        confidence_discrete = np.zeros_like(confidence_continuous, dtype=int)

        # Calculate quantiles for binning
        bin_edges = np.quantile(
            confidence_continuous, np.linspace(0, 1, self.nbins + 1)
        )

        # Check for edge cases in binning
        ## if BOTH the first two AND the last two bin edges are the same, raise an error
        if (
            bin_edges[0] == bin_edges[1]
            and bin_edges[self.nbins - 1] == bin_edges[self.nbins]
        ):
            raise ValueError("Bad bins!")
        ## if ONLY the LAST two bin edges are the same, rebin the confidence values
        elif bin_edges[self.nbins - 1] == bin_edges[self.nbins]:
            # Handle high-confidence issues
            high_confidence = bin_edges[self.nbins]
            bin_edges = np.quantile(
                confidence_continuous[~(confidence_continuous == high_confidence)],
                np.linspace(0, 1, self.nbins),
                method="interpolated_inverted_cdf",
            )

            temp = []
            for b in range(len(bin_edges) - 1):
                temp.append(
                    (confidence_continuous >= bin_edges[b])
                    & (confidence_continuous < bin_edges[b + 1])
                )
            temp.append(confidence_continuous == high_confidence)
            ## save output
            out["confidence_bins"] = np.append(bin_edges, high_confidence)
            out["rebin"] = 1
        ## if ONLY the FIRST two bin edges are the same, rebin the confidence values
        elif bin_edges[0] == bin_edges[1]:
            # Handle low-confidence issues
            low_confidence = bin_edges[0]
            temp = [confidence_continuous == low_confidence]
            bin_edges = np.quantile(
                confidence_continuous[~temp[0]], np.linspace(0, 1, self.nbins)
            )

            for b in range(1, len(bin_edges)):
                temp.append(
                    (confidence_continuous >= bin_edges[b - 1])
                    & (confidence_continuous < bin_edges[b])
                )
            ## save output
            out["confidence_bins"] = np.insert(bin_edges, 0, low_confidence)
            out["rebin"] = 1
        ## if NEITHER the first two nor the last two bin edges are the same, bin the confidence values
        else:
            temp = []
            for b in range(len(bin_edges) - 2):
                temp.append(
                    (confidence_continuous >= bin_edges[b])
                    & (confidence_continuous < bin_edges[b + 1])
                )
            temp.append(
                (confidence_continuous >= bin_edges[-2])
                & (confidence_continuous <= bin_edges[-1])
            )
            ## save output
            out["confidence_bins"] = bin_edges
            out["rebin"] = 0

        # Assign confidence_discrete based on bins and count each bin's size
        for b in range(self.nbins):
            confidence_discrete[temp[b]] = b
            out.setdefault("binned_counts", []).append(np.sum(temp[b]))

        return confidence_discrete, out

    def bin_confidence(self):

        nR_S1 = np.zeros((len(self.ppts), 2 * self.nbins))
        nR_S2 = np.zeros((len(self.ppts), 2 * self.nbins))

        for idx, ppt in enumerate(self.ppts):
            ppt_data = self.data[self.data[self.ppt_identifier] == ppt]
            conf = ppt_data["confidence"]
            assert (conf >= 0).all() and (
                conf <= 1
            ).all(), "Confidence values must be between 0 and 1"
            response = ppt_data["response"]
            stimulus = ppt_data["stimulus"]

            counts = np.zeros((2, 2, self.nbins))

            # Bin confidence values
            confidence_discrete, out = self.new_bin_ratings(conf)
            for S, R in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                i_bin, bincounts = np.unique(
                    confidence_discrete[(stimulus == S) & (response == R)],
                    return_counts=True,
                )
                i_bin = i_bin + self.nbins if R == 1 else self.nbins - i_bin - 1
                if S == 0:
                    nR_S1[idx, i_bin] = bincounts
                else:
                    nR_S2[idx, i_bin] = bincounts

        nR_S1 = nR_S1 + self.adjustment_val if self.apply_adjustment else nR_S1
        nR_S2 = nR_S2 + self.adjustment_val if self.apply_adjustment else nR_S2

        # nR_S1 += self.adjustment_val if self.apply_adjustment else 0
        # nR_S2 += self.adjustment_val if self.apply_adjustment else 0

        if (nR_S1 == 0).any() or (nR_S2 == 0).any():
            raise ValueError("There are bins with zero counts. Please check the data")

        return nR_S1, nR_S2

    def create_df(self):
        """
        Create a pandas dataframe with the observed data for later fitting.
        """
        observed = [
            np.array(
                [
                    self.nR_S1[i, : self.nbins],
                    self.nR_S2[i, : self.nbins],
                    self.nR_S2[i, self.nbins :],
                    self.nR_S1[i, self.nbins :],
                ]
            )
            for i in range(len(self.ppts))
        ]

        df = pd.DataFrame(
            {
                "ppt": self.ppts,
                "nbins": [self.nbins] * len(self.ppts),
                "observed": observed,
            }
        )

        return df

    def rates(self):
        """
        Calculate hit rates and false alarm rates for each confidence bin.

        Returns
        -------
        ratings_hit_rates : numpy.ndarray
            Hit rates for each confidence bin.
        ratings_false_alarms : numpy.ndarray
            False alarm rates for each confidence bin.
        """
        ratings_hit_rates = np.zeros((len(self.ppts), 2 * self.nbins - 1))
        ratings_false_alarms = np.zeros((len(self.ppts), 2 * self.nbins - 1))
        for idx, ppt in enumerate(self.ppts):
            for c in range(1, int(2 * self.nbins)):
                ratings_hit_rates[idx, c - 1] = sum(self.nR_S2[idx, c:]) / sum(
                    self.nR_S2[idx]
                )
                ratings_false_alarms[idx, c - 1] = sum(self.nR_S1[idx, c:]) / sum(
                    self.nR_S1[idx]
                )

        return np.array(ratings_hit_rates), np.array(ratings_false_alarms)

    def plot_ROC(self):
        """
        Plot the ROC curve for each participant.
        """

        for idx, ppt in enumerate(self.ppts):
            fig, axs = plt.subplots()
            axs.scatter(
                self.ratings_false_alarms[idx, : self.nbins],
                self.ratings_hit_rates[idx, : self.nbins],
                label="S2",
            )
            axs.scatter(
                self.ratings_false_alarms[idx, self.nbins - 1 :],
                self.ratings_hit_rates[idx, self.nbins - 1 :],
                label="S1",
            )
            axs.plot([0, 1], [0, 1], "k--")
            axs.set_xlabel("False Alarm Rate")
            axs.set_ylabel("Hit Rate")
            axs.set_title(f"ROC Curve for PPT {ppt}")
            axs.legend()
            plt.show()

    def compute_d_prime(self):
        """
        Calculate d-prime.
        """
        d_prime = np.zeros(len(self.ppts))
        for idx, ppt in enumerate(self.ppts):
            d_prime[idx] = 1 / self.s * self.fninv(
                self.ratings_hit_rates[idx, self.nbins - 1]
            ) - self.fninv(self.ratings_false_alarms[idx, self.nbins - 1])
        return d_prime

    def compute_c1(self):
        """
        Calculate criterion.
        """
        c1 = np.zeros((len(self.ppts), 2 * self.nbins - 1))
        for idx, ppt in enumerate(self.ppts):
            c1[idx] = (
                -1
                / (1 + self.s)
                * (
                    self.fninv(self.ratings_hit_rates[idx])
                    + self.fninv(self.ratings_false_alarms[idx])
                )
            )
        return c1


def meta_model(parameters, **kwargs):
    """
    A function for calculating meta-d based on Signal Detection Theory.

    Parameters
    ----------
    parameters : Parameters
        An object containing the following attributes:
        - nbins : int
            Number of confidence bins.
        - d_prime : float
            d-prime value, also called sensitivity or discriminability index. It is the distance between the two distributions (one for each stimulus type).
        - s : float
            Signal detection criterion.
        - meta_d: float
            meta-d value, also called metacognitive sensitivity.
        - criterion_type1 : float
            Response criteria for stimulus type 1.
        - criterion_type2: array-like
            Response criteria for stimulus type 2.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    dict
        A dictionary containing the calculated probabilities and other various parameters.

    Examples
    --------
    >>> parameters = Parameters(
    >>>    nbins=4,
    >>>    d_prime=2,
    >>>    s=1,
    >>>    meta_d=1.5,
    >>>    criterion_type1=-0.5,
    >>>    criterion_type2=np.array([-1, 0, 1, 1, 1, 0]),
    >>> )
    >>> result = meta_model(parameters)
    >>> print(result)
    {
        "dependent":
            array([[ 0.41163413,  0.58836587,  0.41670862,  0.        ],
        [ 0.12888728,  0.87111272,  2.45572312,  0.        ],
        [ 0.        , -0.367903  ,  0.        , -0.13050534],
        [ 0.        , -0.76099916,  0.        , -1.07448206]]),
        "nbins" : 4,
        "meta_criterion": -0.375,
        "S1mu"  : -0.375,
        "S1sd"  : 1,
        "S2mu"  : 1.125,
        "S2sd"  : 1.0,
    }
    """

    if parameters.s is None:
        s = 1

    nbins = parameters.nbins
    d_prime = parameters.d_prime
    criterion_type1 = parameters.criterion_type1
    criterion_type2 = parameters.criterion_type2
    meta_d = parameters.meta_d
    s = parameters.s
    meta_criterion = meta_d * criterion_type1 / d_prime
    S1mu = -meta_d / 2 - meta_criterion  ## mean for stimulus type 1
    S1sd = 1
    S2mu = meta_d / 2 - meta_criterion  ## mean for stimulus type 2
    S2sd = 1 / s

    if len(criterion_type2) != 2 * nbins - 2:
        raise ValueError(
            "The parameters for the type 2 criteria must be twice the number of confidence bins minus 2"
        )

    ## if criterion_type2 is a cpm.generators.Value object, extract the value
    criterion_type2_array = (
        criterion_type2.value if hasattr(criterion_type2, "value") else criterion_type2
    )
    ## Create criterion array with -inf and inf for the edges
    criterion_type2_array = np.concatenate(
        [[-np.inf], criterion_type2_array, [0], criterion_type2_array, [np.inf]]
    )

    ## calculate the probabilities of hit rates and false alarm rates for each stimulus type
    correct_type_one = norm.cdf(0, S1mu, S1sd)
    incorrect_type_one = norm.cdf(0, S2mu, S2sd)

    correct_type_two = 1 - norm.cdf(0, S2mu, S2sd)
    incorrect_type_two = 1 - norm.cdf(0, S1mu, S1sd)

    # Ensure that the values are clipped to avoid zero values
    correct_type_one = np.clip(correct_type_one, 1e-10, None)
    incorrect_type_one = np.clip(incorrect_type_one, 1e-10, None)
    correct_type_two = np.clip(correct_type_two, 1e-10, None)
    incorrect_type_two = np.clip(incorrect_type_two, 1e-10, None)

    ## calculate probabilities for each confidence bin given the stimulus type and responses
    ## probability_correct_stimulus_one: probability of correct responses given stimulus type 1
    ## probability_incorrect_stimulus_one: probability of incorrect responses given stimulus type 1
    probability_correct_stimulus_one = [
        (
            norm.cdf(criterion_type2_array[i + 1], S1mu, S1sd)
            - norm.cdf(criterion_type2_array[i], S1mu, S1sd)
        )
        / correct_type_one
        for i in np.arange(nbins)
    ]
    probability_incorrect_stimulus_one = [
        (
            norm.cdf(criterion_type2_array[i + 1], S2mu, S2sd)
            - norm.cdf(criterion_type2_array[i], S2mu, S2sd)
        )
        / incorrect_type_one
        for i in np.arange(nbins)
    ]
    ## probability_correct_stimulus_two: probability of correct responses given stimulus type 2
    ## probability_incorrect_stimulus_two: probability of incorrect responses given stimulus type 2
    probability_correct_stimulus_two = [
        (
            (1 - norm.cdf(criterion_type2_array[nbins + i], S2mu, S2sd))
            - (1 - norm.cdf(criterion_type2_array[nbins + i + 1], S2mu, S2sd))
        )
        / correct_type_two
        for i in np.arange(nbins)
    ]
    probability_incorrect_stimulus_two = [
        (
            (1 - norm.cdf(criterion_type2_array[nbins + i], S1mu, S1sd))
            - (1 - norm.cdf(criterion_type2_array[nbins + i + 1], S1mu, S1sd))
        )
        / incorrect_type_two
        for i in np.arange(nbins)
    ]

    probabilities = np.array(
        [
            probability_correct_stimulus_one,
            probability_incorrect_stimulus_one,
            probability_correct_stimulus_two,
            probability_incorrect_stimulus_two,
        ]
    )

    output = {
        "dependent": probabilities,
        "nbins": nbins,
        "meta_criterion": meta_criterion,
        "S1mu": S1mu,
        "S1sd": S1sd,
        "S2mu": S2mu,
        "S2sd": S2sd,
    }

    return output


def InitiateParametersMeta(
    data=None,
    binned_data=None,
    nbins=4,
    s=1,
    apply_adjustment=False,
    adjustment_val=None,
    ppt_identifier="ppt",
    **kwargs,
):
    """
    Initialise parameters for the MetaModel class.

    Parameters
    ----------
    data : pandas.DataFrame
    """
    if data is None and binned_data is None:
        raise ValueError("Either data or binned_data must be provided.")

    helper = MetaHelper(
        data=data,
        nbins=nbins,
        s=s,
        apply_adjustment=apply_adjustment,
        adjustment_val=adjustment_val,
        ppt_identifier=ppt_identifier,
    )

    if data is None:
        helper.nR_S1 = np.array(binned_data["nR_S1"])
        helper.nR_S2 = np.array(binned_data["nR_S2"])
        helper.ratings_hit_rates, helper.ratings_false_alarms = helper.rates()
        helper.d_prime = helper.compute_d_prime()
        helper.c1 = helper.compute_c1()
        helper.criterion_type1 = helper.c1[helper.nbins - 1]
        helper.criterion_type2 = helper.c1 - helper.criterion_type1
        helper.criterion_type2 = np.delete(helper.criterion_type2, helper.nbins - 1)
        helper.data_pandas = helper.create_df()

    params = {
        int(ppt): Parameters(
            meta_d=Value(
                value=helper.d_prime[idx],
                lower=-1.5,
                upper=3.5,
                prior="norm",
                args={"mean": 1, "std": 2},
            ),
            criterion_type2=Value(
                value=helper.criterion_type2[idx],
                lower=-5,
                upper=5,
                prior=multivariate_normal,
                args={
                    "mean": np.delete(np.linspace(-4, 4, 2 * nbins - 1), nbins - 1),
                    "cov": 3 * np.eye(2 * nbins - 2),
                },
            ),
            criterion_type1=helper.criterion_type1[idx],
            s=s,
            d_prime=helper.d_prime[idx],
            nbins=nbins,
        )
        for idx, ppt in enumerate(helper.ppts)
    }

    # Generate matrices A and b based on nbins
    A = np.zeros((2 * nbins - 1, 2 * nbins - 1))
    for i in range(2 * nbins - 3):
        A[i, i + 1 : i + 3] = [-1, 1]
    A[-2, nbins - 1] = -1
    A[-1, nbins] = 1
    b = np.zeros(2 * nbins - 1)

    constraints = LinearConstraint(A=A, b=b)

    return (
        params,
        helper.d_prime,
        helper.criterion_type1,
        helper.data_pandas,
        constraints,
        helper,
    )


class MetaD:

    def __init__(self, data=None, nbins=4, **kwargs):
        self.kwargs = kwargs

        ## initialise parameters
        self.parameters, d1, t1c1, data_pandas, self.constraints, helper = (
            InitiateParametersMeta(
                data=data,
                nbins=nbins,
                apply_adjustment=True,
                ppt_identifier="ppt",
                **self.kwargs,
            )
        )

        self.model = meta_model
        self.data = data
        self.__init_parameters__ = copy.deepcopy(self.parameters)

        # check if parameters is provided as a dictionary with participants as keys
        if not isinstance(self.parameters, Parameters):
            assert isinstance(
                self.parameters, dict
            ), "Parameters must be a dictionary with participants as keys if not a Parameters object."
            assert all(
                [isinstance(p, self.parameters) for p in self.parameters.values()]
            ), "All entries in the dictionary must be Parameters objects."
            self.multiple_ppts = True
            first_ppt = list(self.parameters.keys())[0]
            self.ppt = first_ppt
        else:
            self.multiple_ppts = False
            self.ppt = None

        self.simulation = None
        self.data = data
        # determine the number of trials
        self.__len__, self.__pandas__ = determine_data_length(data)

        self.dependent = {} if self.multiple_ppts else None
        self.parameter_names = (
            list(self.parameters[self.ppt].keys())
            if self.multiple_ppts
            else list(self.parameters.keys())
        )
        self.parameter_sizes = {
            key: (
                self.parameters[self.ppt][key].value.size
                if isinstance(self.parameters[self.ppt][key].value, np.ndarray)
                else 1
            )
            for key in self.parameters[self.ppt].keys()
        }

        self.__run__ = False

    def run(self):
        """
        Run the model.

        Returns
        -------
        None

        """
        ## run the model
        output = (
            self.model(parameters=self.parameters[self.ppt])
            if self.multiple_ppts
            else self.model(parameters=self.parameters)
        )
        self.simulation = output.copy()

        ## update your dependent variables
        if self.multiple_ppts:
            self.dependent[self.ppt] = output.get("dependent").copy()
        else:
            self.dependent = output.get("dependent").copy()

        ## update variables present in both parameters and model output
        self.parameters.update(
            **{
                key: value
                for key, value in output.items()
                if key in self.parameters.keys()
            }
        )

        self.__run__ = True
        return None

    def reset(self, parameters=None, data=None, ppt=None):
        """
        Reset the model.

        Parameters
        ----------
        parameters : dict, array_like, pd.Series or Parameters, optional
            The parameters to reset the model with.

        Notes
        -----
        When resetting the model, and `parameters` is None, reset model to initial state.
        If parameter is `array_like`, it resets the only the parameters in the order they are provided,
        where the last parameter updated is the element in parameters corresponding to len(parameters).

        Examples
        --------
        >>> x = Wrapper(model = mine, data = data, parameters = params)
        >>> x.run()
        >>> x.reset(parameters = [0.1, 1])
        >>> x.run()
        >>> x.reset(parameters = {'alpha': 0.1, 'temperature': 1})
        >>> x.run()
        >>> x.reset(parameters = np.array([0.1, 1, 0.5]))
        >>> x.run()

        Returns
        -------
        None

        """
        self.ppt = ppt if ppt is not None else self.ppt
        if self.__run__:
            self.dependent = {} if self.multiple_ppts else None
            self.simulation = None
            self.parameters = self.__init_parameters__
            self.__run__ = False
        if isinstance(parameters, Parameters):
            self.parameters[self.ppt] = parameters
        # if dict, update using parameters update method
        if isinstance(parameters, dict):
            if isinstance(parameters[self.ppt], Parameters):
                self.parameters[self.ppt].update(**parameters[self.ppt])
            else:
                raise NotImplementedError(
                    "Dictionary update not implemented for MetaSignalDetectionWrapper"
                )
            if isinstance(parameters, pd.Series):
                raise NotImplementedError(
                    "Series update not implemented for MetaSignalDetectionWrapper"
                )
        # if list, update the parameters in for keys in range of 0:len(parameters)
        if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
            offset = 0
            for idx, keys in enumerate(self.parameter_names):
                if self.parameter_sizes[keys] > 1:
                    value = parameters[
                        self.parameter_names.index(keys)
                        + offset : self.parameter_names.index(keys)
                        + offset
                        + self.parameter_sizes[keys]
                    ]
                    offset += self.parameter_sizes[keys] - 1
                else:
                    value = parameters[self.parameter_names.index(keys) + offset]
                (
                    self.parameters[self.ppt].update(**{keys: value})
                    if self.multiple_ppts
                    else self.parameters.update(**{keys: value})
                )
                if idx + offset + 1 == len(parameters):
                    break
        if data is not None:
            self.data = data
            self.__len__, self.__pandas__ = determine_data_length(data)
        return None

    def sample_ppt(self, trials=None, samples=100, ppt=None):
        """ """

        if ppt is None:
            raise ValueError("ppt must be provided for MetaD model")

        stimulus = self.data["stimulus"].values
        stimulus = np.random.randint(2, size=trials)

        d1 = self.parameters[ppt].d1
        t1c1 = self.parameters[ppt].t1c1
        meta_d1 = self.parameters[ppt].meta_d1.value
        t2c1 = self.parameters[ppt].t2c1.value
        nbins = self.parameters[ppt].nbins

        t2c1 = np.concatenate(
            [
                [-np.inf],
                t2c1[: nbins - 1],
                [0],
                t2c1[nbins - 1 :],
                [np.inf],
            ]
        )

        constant_criterion = t1c1 * meta_d1 / d1
        mu_S1 = -meta_d1 / 2 - constant_criterion
        mu_S2 = meta_d1 / 2 - constant_criterion

        s_S1 = norm.rvs(mu_S1, 1, size=(samples, (stimulus == 0).sum()))
        s_S2 = norm.rvs(mu_S2, 1, size=(samples, (stimulus == 1).sum()))

        nR_S1 = np.zeros((samples, 2 * nbins))
        nR_S2 = np.zeros((samples, 2 * nbins))

        for i in range(samples):
            nR_S1[i] = np.histogram(s_S1[i], bins=t2c1)[0]
            nR_S2[i] = np.histogram(s_S2[i], bins=t2c1)[0]

        return nR_S1, nR_S2


# n = 100
# stimID = np.random.randint(0, 2, n)
# response = np.random.randint(0, 2, n)
# rating = np.random.uniform(0, 1, n)
# data = pd.DataFrame(
#     {
#         "stimulus": stimID,
#         "response": response,
#         "confidence": rating,
#     }
# )
# data["ppt"] = "fairy"

# x = MetaHelper(data=data)
# x
# ## bin continuous variables with numpy according to quantiles
# confidence_discrete = np.zeros_like(rating, dtype=int)
# bin_edges = np.quantile(rating, np.linspace(0, 1, x.nbins + 1))

# for b in range(x.nbins):
#     confidence_discrete[(rating >= bin_edges[b]) & (rating < bin_edges[b + 1])] = b
# confidence_discrete[rating == bin_edges[-1]] = x.nbins - 1

# confidence_discrete
# confidence_counts = np.zeros((2, 2, x.nbins), dtype=int)

# for S in range(2):
#     for R in range(2):
#         for b in range(x.nbins):
#             confidence_counts[S, R, b] = np.sum(
#                 (stimID == S) & (response == R) & (confidence_discrete == b)
#             )

# confidence_counts
