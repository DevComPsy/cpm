import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from scipy.optimize import LinearConstraint, minimize

from cpm.generators import Parameters, Value
from cpm.utils.metad import bin_ratings, count_bins
from cpm.core.generators import generate_guesses
from cpm.core.optimisers import objective, numerical_hessian, prepare_data
from cpm.core.data import detailed_pandas_compiler, decompose
from cpm.generators import Simulator, Wrapper
from cpm.core.parallel import detect_cores, execute_parallel

import multiprocess as mp

__all__ = ["metad_generator", "d_prime_calculator", "MetaD", "EstimatorMetaD"]


def metad_generator(parameters, data, **kwargs):
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
            Ratio of standard deviation of the type 1 distribution. If s is not provided, it is set to 1.
        - meta_d: float
            meta-d value, also called metacognitive sensitivity.
        - criterion_type1 : float
            Response criteria for stimulus type 1.
        - criterion_type2: array-like
            Response criteria for stimulus type 2.
    data : object
        An object containing the following attributes:
        - hit_rates : array-like
            Hit rates for each confidence bin.
        - false_alarm_rates : array-like
            False alarm rates for each confidence bin.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    dict
        A dictionary containing the calculated probabilities and other various parameters. See Notes for information on the shape of the probabilities array.

    Notes
    -----
    The output dictionary contains the following keys:
    - dependent : array-like
        A 2D array containing the probabilities of each confidence bin given the stimulus type and responses.
        The first row contains the confidence bin probabilities of correct responses given stimulus type 1.
        The second row contains the confidence bin probabilities of incorrect responses given stimulus type 1.
        The third row contains the confidence bin probabilities of correct responses given stimulus type 2.
        The fourth row contains the confidence bin probabilities of incorrect responses given stimulus type 2.
        Each column is a bins, such that the first column is the first confidence bin, the second column is the second confidence bin, and so on.
    - nbins : int
        Number of confidence bins.
    - meta_criterion : float
        The distance from the mean of the type 1 distribution to the decision criterion.
    - S1mu : float
        Mean for stimulus type 1.
    - S1sd : float
        Standard deviation for stimulus type 1.
    - S2mu : float
        Mean for stimulus type 2.
    - S2sd : float
        Standard deviation for stimulus type 2.


    Examples
    --------
    >>> from cpm.applications.signal_detection import metad_generator
    >>> from cpm.generators import Parameters
    >>> import numpy as np
    >>> # Example usage
    >>> parameters = Parameters(
    >>>    nbins=4,
    >>>    d_prime=2,
    >>>    s=1,
    >>>    meta_d=1.5,
    >>>    criterion_type1=-0.5,
    >>>    criterion_type2=np.array([-1, 0, 1, 1, 1, 0]),
    >>> )
    >>> result = metad_generator(parameters)
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

    s = parameters.s
    if s is None:
        s = 1


    ## non freely varying parameters
    nbins = parameters.bins
    d_prime = parameters.d_prime
    criterion_type1 = (-1 / (1 + s)) * (
        norm.ppf(data.hit_rates) + norm.ppf(data.false_alarm_rates)
    )
    criterion_type1 = criterion_type1[[nbins - 1]]

    ## freely varying parameters
    meta_d = parameters.meta_d
    criterion_type2 = parameters.criterion_type2
    ## meta_criterion is defined to be the distance from the mean of the type 1 distribution to the decision criterion
    ## it is an adjustment we apply to shift the means of the two distributions
    meta_criterion = meta_d * (criterion_type1 / d_prime)

    S1mu = -meta_d / 2 - meta_criterion  ## mean for stimulus type 1
    S1sd = 1  ## standard deviation for stimulus type 1
    S2mu = meta_d / 2 - meta_criterion  ## mean for stimulus type 2
    S2sd = S1sd / s  ## standard deviation for stimulus type 2
    if len(criterion_type2) != 2 * nbins - 2:
        raise ValueError(
            "The parameters for the type 2 criteria must be twice the number of confidence bins minus 2"
        )

    ## if criterion_type2 is a cpm.generators.Value object, extract the value
    criterion_type2_array = criterion_type2.value
    ## Create criterion array with -inf and inf for the edges
    criterion_type2_array = np.concatenate(
        [
            [-np.inf],
            criterion_type2_array[0 : nbins - 1],
            [0],
            criterion_type2_array[nbins - 1 :],
            [np.inf],
        ]
    )

    ## calculate the area under the CDF curve for each stimulus/response combination
    ## auc = area under the curve
    correct_stimulus_one = norm.cdf(0, S1mu, S1sd) ## hit rate
    incorrect_stimulus_one = norm.cdf(0, S2mu, S2sd) ## false alarms

    correct_stimulus_two = 1 - norm.cdf(0, S2mu, S2sd) ## correct rejection
    incorrect_stimulus_two = 1 - norm.cdf(0, S1mu, S1sd) ## misses
    # Ensure that the values are clipped to avoid zero values
    # correct_stimulus_one = np.clip(correct_stimulus_one, 1e-10, None)
    # incorrect_stimulus_one = np.clip(incorrect_stimulus_one, 1e-10, None)
    # correct_stimulus_two = np.clip(correct_stimulus_two, 1e-10, None)
    # incorrect_stimulus_two = np.clip(incorrect_stimulus_two, 1e-10, None)

    ## calculate probabilities for each confidence bin given the stimulus type and responses
    ## probability_correct_stimulus_one: probability of correct stimulus 1 responses (hit rate)
    ## probability_incorrect_stimulus_one: probability of incorrect stimulus 1 responses (misses)
    ## hit rates
    probability_correct_stimulus_one = [
        (
            norm.cdf(criterion_type2_array[i + 1], S1mu, S1sd)
            - norm.cdf(criterion_type2_array[i], S1mu, S1sd)
        )
        / correct_stimulus_one
        for i in np.arange(nbins)
    ]
    ## false alarms
    probability_incorrect_stimulus_one = [
        (
            norm.cdf(criterion_type2_array[i + 1], S2mu, S2sd)
            - norm.cdf(criterion_type2_array[i], S2mu, S2sd)
        )
        / incorrect_stimulus_one
        for i in np.arange(nbins)
    ]
    ## probability_correct_stimulus_two: probability of correct stimulus two responses (correct rejection)
    ## probability_incorrect_stimulus_two: probability of incorrect stimulus 2 responses (false alarms)
    ## correct rejection
    probability_correct_stimulus_two = [
        (
            (1 - norm.cdf(criterion_type2_array[nbins + i], S2mu, S2sd))
            - (1 - norm.cdf(criterion_type2_array[nbins + i + 1], S2mu, S2sd))
        )
        / correct_stimulus_two
        for i in np.arange(nbins)
    ]
    ## false alarms
    probability_incorrect_stimulus_two = [
        (
            (1 - norm.cdf(criterion_type2_array[nbins + i], S1mu, S1sd))
            - (1 - norm.cdf(criterion_type2_array[nbins + i + 1], S1mu, S1sd))
        )
        / incorrect_stimulus_two
        for i in np.arange(nbins)
    ]
    ## the probabilities below are ordered to correspond with the human data
    ## see cpm.utils.metad count_bins function on how participant data is ordered
    probabilities = np.array(
        [
            np.asarray(probability_correct_stimulus_one).flatten(),  ## hit rates
            np.asarray(probability_incorrect_stimulus_one).flatten(),  ## false alarms
            np.asarray(probability_correct_stimulus_two).flatten(),  ## correct rejections
            np.asarray(probability_incorrect_stimulus_two).flatten(),  ## misses
        ],
    )
    output = {
        "dependent": probabilities,
        "bins": nbins.value,
        "type_2_criterion_array": criterion_type2_array,
        "meta_criterion": meta_criterion,
        "meta_d": meta_d.value,
        "S1mu": S1mu,
        "S1sd": S1sd,
        "S2mu": S2mu,
        "S2sd": S2sd,
        "criterion_type1": criterion_type1,
        "hits_and_false_alarms": data,
        "area_under_the_curve" : np.array([
            correct_stimulus_one,
            incorrect_stimulus_one,
            correct_stimulus_two,
            incorrect_stimulus_two,
        ])
    }

    return output


def d_prime_calculator(data, s=1):
    """
    A function that calculates d-prime from hit rates and false alarms.

    ```
    d' = z(HR) - z(FAR)
    ```

    where HR is the hit rate (true positive rate) and FAR is the false alarm rate (false positive rate).
    z is the inverse of the cumulative distribution function of the standard normal distribution.
    """
    dprime = (1 / s) * norm.ppf(data.hit_rates) - norm.ppf(data.false_alarm_rates)

    output = {
        "d_prime": dprime,
        "hit_rates": data.hit_rates,
        "false_alarm_rates": data.false_alarm_rates,
    }
    return output


class MetaD:

    def __init__(self, data=None, dimensions=2, bins=4, parameters=None):
        """
        Parameters
        ----------

        data : pandas.DataFrame
            A pandas.DataFrame containing the following columns: ["stimulus", "response", "confidence"]. The "stimulus" column contains the stimulus identifier, the "response" column contains the response identifier, and the "confidence" column contains the confidence rating. The "confidence" column should be a continuous variable. Counts start from 0, so if the first stimulus is identified as 1, adjust it to be 0.
        """
        ## if any of these values are none, then throw error
        if data is None:
            raise ValueError("Data is required to calculate meta-d")
        ## check if data is a pandas dataframe
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame, see documentation.")
        self.data = data
        self.dimensions = dimensions
        self.bins = bins
        if parameters is None:
            parameters = np.concatenate(
                [[1], np.random.uniform(-5, 5, size=(dimensions * bins) - 2)]
            ).flatten()

        self.parameters = Parameters(
            meta_d=Value(
                value=parameters[0],
                lower=-10,
                upper=10,
                prior="norm",
                args={"mean": 1, "std": 2},
            ),
            criterion_type2=Value(
                value=parameters[1:],
                lower=np.array([[-20] * (bins - 1), [0] * (bins - 1)]).flatten(),
                upper=np.array([[0] * (bins - 1), [20] * (bins - 1)]).flatten(),
                prior=multivariate_normal,
                args={
                    "mean": np.delete(np.linspace(-4, 4, 2 * bins - 1), bins - 1),
                    "cov": 3 * np.eye(2 * bins - 2),
                },
            ),
            s=1,
            d_prime=1,
            bins=bins,
        )
        self.bounds = self.parameters.bounds()

        ## Here, we constrain the parameters to ensure that the criteria are in the correct order
        ## this means that criteria for bin i is always larger than i + 1
        ## this is different from other toolboxes. because they order the criteria for different responses in opposing orders
        ## here, we always have the criteria for all responses in the same order
        nCriteria = int(2 * self.bins - 1)
        num_rows = nCriteria - 2
        lb = np.full(nCriteria - 2, 1e-5)
        ub = np.full(nCriteria - 2, np.inf)

        A = np.eye(num_rows, k=0) - np.eye(num_rows, k=1)
        A = np.hstack((np.zeros((num_rows, 1)), A, np.zeros((num_rows, 1))))
        A[-1, -1] = -1

        self.constraints = LinearConstraint(A=A, lb=lb, ub=ub)

        self.d_prime_formula = d_prime_calculator
        self.meta_d_formula = metad_generator

    def run(self):
        """
        Run the meta-d analysis.
        """
        ## type 1 sensitivity
        ## fist, calculate hit rates and false alarm rates
        aggregate = self.data.groupby(["stimulus", "response"]).count().reset_index()
        ## calculate hit rates for when stimulus == response
        hit_rates = (
            aggregate[
                (aggregate.stimulus == aggregate.response)
            ].confidence.sum()
            / aggregate.confidence.sum()
        )
        false_alarm_rates = (
            aggregate[
                (aggregate.stimulus != aggregate.response)
            ].confidence.sum()
            / aggregate.confidence.sum()
        )
        transformed_type1 = pd.Series(
            {
                "hit_rates": hit_rates,
                "false_alarm_rates": false_alarm_rates,
            }
        )
        self.d_prime = pd.Series(self.d_prime_formula(transformed_type1))
        self.parameters.update(
            **{
                "d_prime": self.d_prime.d_prime,
            }
        )
        ## type 2 sensitivity
        discretised_data = bin_ratings(
            ratings=self.data.confidence, xtdout=False, nbins=self.bins
        )
        counted_data = count_bins(
            stimulus_identifier=self.data.stimulus.astype(int),
            response=self.data.response.astype(int),
            n_ratings=4,
            rating=discretised_data,
            pandas=True,
        )
        ## calculate counts of responses for each bin
        ## we use it to calculate the meta-criterion
        ## to adjust means of distributions to avoid problems with optimisation
        hit_rates_binned = counted_data[counted_data.stimulus == 1].counts.to_numpy()
        hit_rates_binned /= np.sum(hit_rates_binned)
        hit_rates_binned = hit_rates_binned[::-1]
        hit_rates_binned = hit_rates_binned[1:]
        false_alarm_rates_binned = counted_data[counted_data.stimulus == 0].counts.to_numpy()
        false_alarm_rates_binned /= np.sum(false_alarm_rates_binned)
        false_alarm_rates_binned = false_alarm_rates_binned[::-1]
        false_alarm_rates_binned = false_alarm_rates_binned[1:]
        
        transformed_type2 = pd.Series(
            {
                "hit_rates": hit_rates_binned,
                "false_alarm_rates": false_alarm_rates_binned,
            }
        )
        self.meta_d = pd.Series(self.meta_d_formula(self.parameters, transformed_type2))
        self.dependent = self.meta_d.get("dependent").flatten()
        return None

    def reset(self, parameters=None, data=None):
        """
        Update the parameters of the model.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters to update.
        """
        if parameters is not None:
            self.parameters.update(
                **{"meta_d": parameters[0], "criterion_type2": parameters[1:]}
            )
        if data is not None:
            self.data = data
        return None

    def export(self):
        """
        Export model output in a pandas DataFrame
        """
        output = pd.concat(
            [self.meta_d.drop("dependent").to_frame().T, self.d_prime.to_frame().T],
            axis=1,
        )
        return output


class EstimatorMetaD:
    """
    Class representing scipy's Minimize algorithm wrapped for subject-level parameter estimations.

    Parameters
    ----------
    model : cpm.generators.Wrapper
        The model to be optimized.
    data : pd.DataFrame, pd.DataFrameGroupBy, list
        The data used for optimization. If a pd.Dataframe, it is grouped by the `ppt_identifier`. If it is a pd.DataFrameGroupby, groups are assumed to be participants. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    minimisation : function
        The loss function for the objective minimization function. See the `minimise` module for more information. User-defined loss functions are also supported.
    number_of_starts : int
        The number of random initialisations for the optimization. Default is `1`.
    initial_guess : list or array-like
        The initial guess for the optimization. Default is `None`. If `number_of_starts` is set, and the `initial_guess` parameter is 'None', the initial guesses are randomly generated from a uniform distribution.
    parallel : bool
        Whether to use parallel processing. Default is `False`.
    cl : int
        The number of cores to use for parallel processing. Default is `None`. If `None`, the number of cores is set to 2.
        If `cl` is set to `None` and `parallel` is set to `True`, the number of cores is set to the number of cores available on the machine.
    libraries : list, optional
        The libraries to import for parallel processing for `ipyparallel` with the IPython kernel. Default is `["numpy", "pandas"]`
    ppt_identifier : str
        The key in the participant data dictionary that contains the participant identifier. Default is `None`. Returned in the optimization details.
    **kwargs : dict
        Additional keyword arguments. See the [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) documentation for what is supported.


    Notes
    -----
    The data parameter must contain all input to the model, including the observed data. The data parameter can be a pandas DataFrame, a pandas DataFrameGroupBy object, or a list of dictionaries. If the data parameter is a pandas DataFrame, it is assumed that the data needs to be grouped by the participant identifier, `ppt_identifier`. If the data parameter is a pandas DataFrameGroupBy object, the groups are assumed to be participants. If the data parameter is a list of dictionaries, each dictionary should contain the data for a single participant, including information about the experiment and the results. The observed data for each participant should be included in the dictionary under the key or column 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable calculated by the model Wrapper.

    The optimization process is repeated `number_of_starts` times, and only the best-fitting output from the best guess is stored.
    """

    def __init__(
        self,
        model=None,
        data=None,
        initial_guess=None,
        minimisation=None,
        method="Nelder-Mead",
        cl=None,
        parallel=False,
        libraries=["numpy", "pandas"],
        prior=False,
        number_of_starts=1,
        ppt_identifier=None,
        display=False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        if not isinstance(self.model, MetaD):
            raise ValueError("The model must be an instance of MetaD.")
        self.data = data
        self.ppt_identifier = ppt_identifier
        self.data, self.participants, self.groups, self.__pandas__ = prepare_data(
            data, self.ppt_identifier
        )

        self.loss = minimisation
        self.prior = prior
        self.kwargs = kwargs
        self.display = display
        self.prior = prior

        self.method = method
        self.parameter_names = [
            "meta_d",
            "criterion1_type2",
            "criterion2_type2",
            "criterion3_type3",
            "criterion4_type4",
            "criterion5_type5",
            "criterion6_type6",
        ]

        self.fit = []
        self.details = []
        self.parameters = []

        self.initial_guess = generate_guesses(
            bounds=self.model.parameters.bounds(),
            number_of_starts=number_of_starts,
            guesses=initial_guess,
            shape=(number_of_starts, 7),
        )

        self.__parallel__ = parallel
        self.__libraries__ = libraries
        self.__current_guess__ = self.initial_guess[0]

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = detect_cores()

    def optimise(self):
        """
        Performs the optimization process.

        Returns:
        - None
        """

        def __unpack(x, id=None):
            keys = ["x", "fun", "nit", "nfev", "status", "success", "message"]
            if id is not None:
                keys.append(id)
            out = {}
            for i in range(len(keys)):
                out[keys[i]] = x[keys[i]]
            return out

        def __task(participant, **args):

            participant_dc, _, ppt = decompose(
                participant=participant,
                pandas=self.__pandas__,
                identifier=self.ppt_identifier,
            )

            discrete = bin_ratings(
                ratings=participant_dc.confidence, xtdout=False, nbins=self.model.bins
            )
            observed = count_bins(
                stimulus_identifier=participant_dc.stimulus,
                response=participant_dc.response,
                n_ratings=4,
                rating=discrete,
                pandas=False,
            ).flatten()

            model.reset(data=participant_dc)

            result = minimize(
                objective,
                x0=self.__current_guess__,
                args=(model, observed, loss, prior),
                method=self.method,
                **self.kwargs,
            )
            result = {**result}

            def f(x):
                return objective(x, model, observed, loss, prior)

            hessian = numerical_hessian(func=f, params=result["x"] + 1e-3)

            result.update({"hessian": hessian})
            # if participant data contains identifiers, return the identifiers too
            result.update({"ppt": ppt})
            return result

        def __extract_nll(result):
            output = np.zeros(len(result))
            for i in range(len(result)):
                output[i] = result[i]["fun"]
            return output.copy()

        bounds = self.model.parameters.bounds()
        bounds = np.asarray(bounds).T
        bounds = list(map(tuple, bounds))
        self.kwargs.update({"bounds": bounds})

        loss = self.loss
        model = self.model
        prior = self.prior

        for i in range(len(self.initial_guess)):
            print(
                f"Starting optimization {i+1}/{len(self.initial_guess)} from {self.initial_guess[i]}"
            )
            self.__current_guess__ = self.initial_guess[i]
            if self.__parallel__:
                results = execute_parallel(
                    job=__task,
                    data=self.data,
                    method=None,
                    cl=self.cl,
                    pandas=self.__pandas__,
                    libraries=self.__libraries__,
                )
            else:
                results = list(map(__task, self.data))

            ## extract the negative log likelihoods for each ppt
            if i == 0:
                old_nll = __extract_nll(results)
                self.details = copy.deepcopy(results)
                parameters = {}
                for result in results:
                    for i in range(len(self.parameter_names)):
                        parameters[self.parameter_names[i]] = copy.deepcopy(
                            result["x"][i]
                        )
                    self.parameters.append(copy.deepcopy(parameters))
                    self.fit.append(
                        __unpack(copy.deepcopy(result), id=self.ppt_identifier)
                    )
            else:
                nll = __extract_nll(results)
                # check if ppt fit is better than the previous fit
                indices = np.where(nll < old_nll)[0]
                for ppt in indices:
                    self.details[ppt] = copy.deepcopy(results[ppt])
                    for i in range(len(self.parameter_names)):
                        self.parameters[ppt][self.parameter_names[i]] = copy.deepcopy(
                            results[ppt]["x"][i]
                        )
                    self.fit[ppt] = __unpack(
                        copy.deepcopy(results[ppt]), id=self.ppt_identifier
                    )

        return None

    def reset(self, initial_guess=True):
        """
        Resets the optimization results and fitted parameters.

        Parameters
        ----------
        initial_guess : bool, optional
            Whether to reset the initial guess (generates a new set of random numbers within parameter bounds). Default is `True`.

        Returns
        -------
        None
        """
        self.fit = []
        self.details = []
        self.parameters = []
        if initial_guess:
            self.initial_guess = generate_guesses(
                bounds=self.model.parameters.bounds(),
                number_of_starts=self.initial_guess.shape[0],
                guesses=None,
                shape=self.initial_guess.shape,
            )
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
