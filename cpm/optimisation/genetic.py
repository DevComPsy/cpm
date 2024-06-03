from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
import copy
import warnings
import multiprocess as mp


from . import minimise
from . import utils
from ..generators import Simulator, Wrapper


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


class DifferentialEvolution:
    """
    Class representing the Differential Evolution optimization algorithm.

    Parameters
    ----------
    model : object
        The model to be optimized.
    bounds : object
            The parameter bounds for the optimization. The bounds should be a list of tuples, where each tuple contains the lower and upper bounds for a parameter. Elements of a tuple must corespond to the parameters in Parameters. If less bounds are provided than parameters, the algorithm will only fit those.
    data : object
        The data used for optimization. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    minimisation : function
        The loss function for the objective minimization function. Default is `minimise.LogLikelihood.bernoulli`. See the `minimise` module for more information. User-defined loss functions are also supported, but they must conform to the format of currently implemented ones.
    **kwargs : dict
        Additional keyword arguments. See the [`scipy.optimize.differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) documentation for what is supported.

    Attributes
    ----------
    function : Wrapper
        The model object.
    data : object
        The data used for optimization. An array of dictionaries, where each dictionary contains the data for a single participant.
    loss : function
        The loss function for the objective minimization function.
    kwargs : dict
        Additional keyword arguments.
    fit : list
        List to store the optimization results. It includes the best-fitting parameters and the objective function value.
    details : list
        List to store the optimization details. It includes all information returned by the optimization algorithm in addition to what is already stored in `fit`.
    parameters : list
        List to store the best-fitting parameters as dictionaries.
    participant : object
        The current participant data.
    parameter_names : list
        The names of the model parameters.
    bounds : object
        The parameter bounds for the optimization.

    Notes
    -----
    The `data` parameter is an array of dictionaries, where each dictionary contains the data for a single participant. The dictionary should contain the keys needed to simulate behaviour using the model, such as trials and feedback. The dictionary should also contain the observed data for the participant, titled 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable the model `Wrapper`.
    """

    def __init__(
        self,
        model=None,
        data=None,
        minimisation=minimise.LogLikelihood.bernoulli,
        parallel=False,
        cl=None,
        prior=False,
        ppt_identifier=None,
        display=False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.loss = minimisation
        self.kwargs = kwargs
        self.fit = []
        self.details = []
        self.parameters = []
        self.participant = data[0]
        self.display = display
        self.ppt_identifier = ppt_identifier
        self.prior = prior

        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameters.free()
        if isinstance(model, Simulator):
            raise ValueError(
                "The DifferentialEvolution algorithm is not compatible with the Simulator object."
            )

        self.__parallel__ = parallel

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = mp.cpu_count()

        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameters.free()
            bounds = self.model.parameters.bounds()
            bounds = np.asarray(bounds).T
            bounds = list(map(tuple, bounds))
            self.bounds = bounds
        if isinstance(model, Simulator):
            raise ValueError(
                "The DifferentialEvolution algorithm is not compatible with the Simulator object."
            )

    def optimise(self):
        """
        Performs the optimization process.

        Returns:
        - None
        """

        def __task(participant, **args):
            model.data = participant
            result = differential_evolution(
                func=minimum,
                bounds=self.bounds,
                args=((model, participant.get("observed"), loss, prior)),
                **self.kwargs,
            )
            if self.ppt_identifier is not None:
                result.ppt = participant.get(self.ppt_identifier)
            return result

        loss = self.loss
        model = self.model
        prior = self.prior

        if self.__parallel__:
            with mp.Pool(self.cl) as pool:
                results = pool.map(__task, self.data)
        else:
            results = list(map(__task, self.data))

        self.details = copy.deepcopy(results)
        for result in results:
            self.parameters.append(
                copy.deepcopy(
                    utils.extract_params_from_fit(
                        data=result.x, keys=self.parameter_names
                    )
                )
            )
            self.fit.append({"parameters": result.x, "fun": copy.deepcopy(result.fun)})

        return None

    def reset(self):
        """
        Resets the optimization results and fitted parameters.

        Returns:
        - None
        """
        self.fit = []
        self.parameters = []
        return None

    def export(self, details=False):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Parameters
        ----------
        details : bool
            Whether to include the various metrics related to the optimisation routine in the output.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters. If `details` is `True`, the DataFrame will also include the optimization details.

        Notes
        -----
        The DataFrame will not contain the population and population_energies keys from the optimization details.
        If you want to investigate it, please use the `details` attribute.
        """
        ranged = len(self.parameter_names)
        output = pd.DataFrame()
        for i in range(len(self.fit)):
            current = pd.DataFrame(self.fit[i]["parameters"]).T
            current.columns = self.parameter_names[0 : len(current.columns)]
            current["fun"] = self.fit[i]["fun"]
            output = pd.concat([output, current], axis=0)

        if details:
            metrics = utils.detailed_pandas_compiler(
                self.details, method="differential_evolution"
            )
            output.reset_index(drop=True, inplace=True)
            metrics.reset_index(drop=True, inplace=True)
            output = pd.concat([output, metrics], axis=1)
        return output
