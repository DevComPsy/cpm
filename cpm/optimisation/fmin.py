from scipy.optimize import fmin
from . import minimise
from . import utils
from ..generators import Simulator, Wrapper

import numpy as np
import pandas as pd
import copy
import dask


def minimum(pars, function, data, loss, **args):
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
    observed = data
    metric = loss(predicted, observed, **args)
    del predicted, observed
    if metric == float("inf") or metric == float("-inf") or metric == float("nan"):
        metric = 1e10
    return metric


class Fmin:
    """
    Class representing the Fmin search optimization algorithm.

    Parameters
    ----------
    model : object
        The model to be optimized.
    data : object
        The data used for optimization. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    loss : function
        The loss function for the objective minimization function. Default is `minimise.LogLikelihood.continuous`. See the `minimise` module for more information. User-defined loss functions are also supported.
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


    Notes
    -----
    The `data` parameter is an array of dictionaries, where each dictionary contains the data for a single participant. The dictionary should contain the keys needed to simulate behaviour using the model, such as trials and feedback. The dictionary should also contain the observed data for the participant, titled 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable the model `Wrapper`.
    """

    def __init__(
        self,
        model=None,
        data=None,
        initial_guess=None,
        minimisation=minimise.LogLikelihood.continuous,
        **kwargs
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.loss = minimisation
        self.initial_guess = initial_guess
        self.kwargs = kwargs
        self.fit = []
        self.details = []
        self.parameters = []
        self.participant = data[0]
        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameter_names
        if isinstance(model, Simulator):
            raise ValueError(
                "The Fmin algorithm is not compatible with the Simulator object."
            )
        self.auxiliary = {
            "n": len(self.participant.get("observed")),
            "k": len(self.initial_guess),
        }

    # def minimise(self, pars, **args):
    #     """
    #     The `minimise` function calculates a metric by comparing predicted values with
    #     observed values.

    #     Parameters
    #     ----------
    #     pars
    #         The `pars` parameter is a dictionary that contains the parameters for the
    #         function that needs to be minimized.

    #     Returns
    #     -------
    #         The metric value is being returned.

    #     """
    #     evaluated = copy.deepcopy(self.function)
    #     evaluated.reset(pars)
    #     evaluated.run()
    #     predicted = evaluated.dependent
    #     observed = self.participant.get("observed")
    #     metric = self.loss(predicted, observed, **self.auxiliary)
    #     del predicted, observed
    #     if metric == float("inf") or metric == float("-inf") or metric == float("nan"):
    #         metric = 1e10
    #     return metric

    def __single(self, do):

        return None

    def optimise(self):
        """
        Performs the optimization process.

        Returns:
        - None
        """

        def __unpack(x):
            keys = ["xopt", "fopt", "iter", "funcalls", "warnflag"]
            out = {}
            for i in range(len(keys)):
                out[keys[i]] = x[i]
            return out

        def __task(model, participant, loss, **args):
            result = fmin(
                minimum,
                x0=self.initial_guess,
                args=(model, participant.get("observed"), loss),
                disp=False,
                **self.kwargs,
                full_output=True
            )
            return result

        from dask.distributed import Client, progress

        client = Client(threads_per_worker=4, n_workers=1)

        # # Submit work to happen in parallel
        # results = []
        # for filename in filenames:
        #     data = client.submit(load, filename)
        #     result = client.submit(process, data)
        #     results.append(result)

        # # Gather results back to local computer
        # results = client.gather(results)
        distribution = []
        for i in range(len(self.data)):
            self.participant = self.data[i]
            self.model.data = self.participant
            tt = client.submit(__task)(
                self.model,
                self.participant,
                self.loss,
            )
            distribution.append(tt)
        self.details = distribution
        # self.details = client.gather(distribution)
        # self.details = dask.compute(*distribution)

        # add the parameters to the list
        # result = __task(self.model, self.participant, self.loss, **self.auxiliary)
        # self.details.append(__unpack(copy.deepcopy(result)))
        # parameters = {}
        # for i in range(len(self.initial_guess)):
        #     parameters[self.parameter_names[i]] = result[0][i]
        # self.parameters.append(parameters)
        # # add the results to the list
        # self.fit.append({"parameters": result[0], "fun": copy.deepcopy(result[1])})
        return None

    def reset(self):
        """
        Resets the optimization results and fitted parameters.

        Returns:
            None
        """
        self.fit = []
        self.parameters = []
        return None

    def export(self):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters.
        """
        output = utils.detailed_pandas_compiler(self.details)
        output.reset_index(drop=True, inplace=True)
        return output
