from . import minimise
from . import utils
from ..generators import Simulator, Wrapper

from scipy.optimize import minimize
import numpy as np
import pandas as pd
import copy
import warnings
import multiprocess as mp
import numdifftools as nd


def minimum(pars, function, data, loss, prior=False, **args):
    """
    The `minimise` function calculates a metric by comparing predicted values with
    observed values.

    Parameters
    ----------
    pars
        The `pars` parameter is a dictionary that contains the parameters for the
        function that needs to be minimized.
    function
        The `function` parameter is the function that needs to be minimized.
    data
        The `data` parameter is the data that is used to compare the predicted values
        with the observed values.
    loss
        The `loss` parameter is the loss function that is used to calculate the metric
        value.
    args
        The `args` parameter is a dictionary that contains additional parameters that
        are used in the loss function.

    Returns
    -------
        The metric value is being returned.

    """
    function.reset(pars)
    function.run()
    predicted = function.dependent
    observed = copy.deepcopy(data)
    metric = loss(predicted, observed, **args)
    del predicted, observed
    if np.isnan(metric) or np.isinf(metric):
        metric = 1e10
    if prior:
        prior_pars = function.parameters.PDF(log=True)
        metric += -prior_pars
    return metric


class Minimize:
    """
    Class representing scipy's Minimize algorithm wrapped for subject-level parameter estimations.

    Parameters
    ----------
    model : object
        The model to be optimized.
    data : object
        The data used for optimization. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
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
    ppt_identifier : str
        The key in the participant data dictionary that contains the participant identifier. Default is `None`. Returned in the optimization details.
    **kwargs : dict
        Additional keyword arguments. See the [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) documentation for what is supported.


    Notes
    -----
    The `data` parameter is an array of dictionaries, where each dictionary contains the data for a single participant. The dictionary should contain the keys needed to simulate behaviour using the model, such as trials and feedback. The dictionary **MUST** also contain the observed data for the participant, titled 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable calculated by the model `Wrapper`.

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
        prior=False,
        number_of_starts=1,
        ppt_identifier=None,
        display=False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.loss = minimisation
        self.prior = prior
        self.kwargs = kwargs
        self.participant = data[0]
        self.display = display
        self.ppt_identifier = ppt_identifier
        self.method = method
        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameters.free()
        if isinstance(model, Simulator):
            raise ValueError(
                "The GradientFree class is not compatible with the Simulator object."
            )

        self.fit = []
        self.details = []
        self.parameters = []

        if number_of_starts is not None and initial_guess is not None:
            ## convert to a 2D array
            initial_guess = np.asarray(initial_guess)
            if len(initial_guess.shape) == 1:
                initial_guess = np.expand_dims(initial_guess, axis=0)
            ## assign the initial guess and raise an error if the number of starts does not match the number of initial guesses
            self.initial_guess = initial_guess
            if np.asarray(initial_guess).shape[0] != number_of_starts:
                raise ValueError(
                    "The number of initial guesses must match the number of starts."
                )

        if number_of_starts is not None and initial_guess is None:
            bounds = self.model.parameters.bounds()
            self.initial_guess = np.random.uniform(
                low=bounds[0],
                high=bounds[1],
                size=(number_of_starts, len(self.parameter_names)),
            )

        self.__parallel__ = parallel
        self.__current_guess__ = self.initial_guess[0]

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = mp.cpu_count()

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

            model.reset(data=participant)

            result = minimize(
                minimum,
                x0=self.__current_guess__,
                args=(model, participant.get("observed"), loss, prior),
                method=self.method,
                **self.kwargs,
            )
            result = {**result}
            hessian = Hessian(result["x"], model, participant.get("observed"), loss)
            result.update({"hessian": hessian})
            # if participant data contains identifiers, return the identifiers too
            if self.ppt_identifier is not None:
                result.update({"ppt": participant.get(self.ppt_identifier)})
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
        Hessian = nd.Hessian(minimum)

        for i in range(len(self.initial_guess)):
            print(
                f"Starting optimization {i+1}/{len(self.initial_guess)} from {self.initial_guess[i]}"
            )
            self.__current_guess__ = self.initial_guess[i]
            if self.__parallel__:
                with mp.Pool(self.cl) as pool:
                    results = pool.map(__task, self.data)
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
            bounds = self.model.parameters.bounds()
            self.initial_guess = np.random.uniform(
                low=bounds[0], high=bounds[1], size=self.initial_guess.shape
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
        output = utils.detailed_pandas_compiler(self.fit)
        output.reset_index(drop=True, inplace=True)
        return output
