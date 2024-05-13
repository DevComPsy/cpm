from . import minimise
from . import utils
from ..generators import Simulator, Wrapper

from scipy.optimize import fmin, fmin_l_bfgs_b
import numpy as np
import pandas as pd
import copy
import multiprocess as mp
import numdifftools as nd

__all__ = ["Fmin", "FminBound"]


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
    if metric == float("inf") or metric == float("-inf") or metric == float("nan"):
        metric = 1e10
    if prior:
        prior_pars = function.parameters.PDF(log=True)
        metric += -prior_pars
    return metric


class Fmin:
    """
    Class representing the Fmin search (unbounded) optimization algorithm using a downhill simplex.

    Parameters
    ----------
    model : object
        The model to be optimized.
    data : object
        The data used for optimization. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    minimisation : function
        The loss function for the objective minimization function. See the `minimise` module for more information. User-defined loss functions are also supported.
    parallel : bool
        Whether to use parallel processing. Default is `False`.
    cl : int
        The number of cores to use for parallel processing. Default is `None`. If `None`, the number of cores is set to 2.
        If `cl` is set to `None` and `parallel` is set to `True`, the number of cores is set to the number of cores available on the machine.
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
    The `data` parameter is an array of dictionaries, where each dictionary contains the data for a single participant. The dictionary should contain the keys needed to simulate behaviour using the model, such as trials and feedback. The dictionary **MUST** also contain the observed data for the participant, titled 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable calculated by the model `Wrapper`.
    """

    def __init__(
        self,
        model=None,
        data=None,
        initial_guess=None,
        minimisation=None,
        cl=None,
        parallel=False,
        prior=False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.loss = minimisation
        self.initial_guess = initial_guess
        self.prior = prior
        self.kwargs = kwargs
        self.fit = []
        self.details = []
        self.parameters = []
        self.participant = data[0]
        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameters.free()
        if isinstance(model, Simulator):
            raise ValueError(
                "The Fmin algorithm is not compatible with the Simulator object."
            )
        self.auxiliary = {
            "n": len(self.participant.get("observed")),
            "k": len(self.initial_guess),
        }
        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = mp.cpu_count()
        if cl is None and not parallel:
            self.cl = 2

    def optimise(self):
        """
        Performs the optimization process.

        Returns:
        - None
        """

        def __unpack(x):
            keys = ["xopt", "fopt", "iter", "funcalls", "warnflag", "hessian"]
            out = {}
            for i in range(len(keys)):
                out[keys[i]] = x[i]
            return out

        def __task(participant, **args):
            result = fmin(
                minimum,
                x0=self.initial_guess,
                args=(model, participant.get("observed"), loss, prior),
                disp=False,
                **self.kwargs,
                full_output=True,
            )
            hessian = Hessian(result[0], model, participant.get("observed"), loss)
            result = (*result, hessian)
            return result

        loss = self.loss
        model = self.model
        prior = self.prior
        Hessian = nd.Hessian(minimum)

        pool = mp.Pool(self.cl)
        results = pool.map(__task, self.data)
        pool.close()
        del pool
        self.details = results

        parameters = {}
        for result in results:
            for i in range(len(self.initial_guess)):
                parameters[self.parameter_names[i]] = result[0][i]
            self.parameters.append(parameters)
            self.fit.append(__unpack(result))

        return None

    def reset(self):
        """
        Resets the optimization results and fitted parameters.

        Returns:
            None
        """
        self.fit = []
        self.details = []
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
        output = utils.detailed_pandas_compiler(self.fit)
        output.reset_index(drop=True, inplace=True)
        return output


class FminBound:
    """
    Class representing the Fmin search (bounded) optimization algorithm using the L-BFGS-B method.

    Parameters
    ----------
    model : object
        The model to be optimized.
    data : object
        The data used for optimization. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    minimisation : function
        The loss function for the objective minimization function. See the `minimise` module for more information. User-defined loss functions are also supported.
    parallel : bool
        Whether to use parallel processing. Default is `False`.
    cl : int
        The number of cores to use for parallel processing. Default is `None`. If `None`, the number of cores is set to 2.
        If `cl` is set to `None` and `parallel` is set to `True`, the number of cores is set to the number of cores available on the machine.
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
    The `data` parameter is an array of dictionaries, where each dictionary contains the data for a single participant. The dictionary should contain the keys needed to simulate behaviour using the model, such as trials and feedback. The dictionary **MUST** also contain the observed data for the participant, titled 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable calculated by the model `Wrapper`.
    """

    def __init__(
        self,
        model=None,
        data=None,
        initial_guess=None,
        minimisation=None,
        cl=None,
        parallel=False,
        prior=False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.loss = minimisation
        self.initial_guess = initial_guess
        self.prior = prior
        self.kwargs = kwargs
        self.fit = []
        self.details = []
        self.parameters = []
        self.participant = data[0]
        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameters.free()
        if isinstance(model, Simulator):
            raise ValueError(
                "The Fmin algorithm is not compatible with the Simulator object."
            )
        self.auxiliary = {
            "n": len(self.participant.get("observed")),
            "k": len(self.initial_guess),
        }
        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = mp.cpu_count()
        if cl is None and not parallel:
            self.cl = 2

    def optimise(self):
        """
        Performs the optimization process.

        Returns:
        - None
        """

        def __unpack(x):
            keys = ["x", "f", "grad", "task", "funcalls", "nit", "warnflag", "hessian"]
            out = {}
            for i in range(len(keys)):
                out[keys[i]] = x[i]
            out["fopt"] = out.pop("f")
            return out

        bounds = self.model.parameters.bounds()
        bounds = tuple(map(tuple, bounds))
        loss = self.loss
        model = self.model
        prior = self.prior
        Hessian = nd.Hessian(minimum)

        def __task(participant, **args):
            result = fmin_l_bfgs_b(
                minimum,
                x0=self.initial_guess,
                bounds=bounds,
                args=(model, participant.get("observed"), loss, prior),
                disp=False,
                **self.kwargs,
            )
            hessian = Hessian(result[0], model, participant.get("observed"), loss)
            result = (*result[0:2], *tuple(list(result[2].values())), hessian)
            return result

        pool = mp.Pool(self.cl)
        results = pool.map(__task, self.data)
        pool.close()
        del pool
        self.details = results

        parameters = {}
        for result in results:
            for i in range(len(self.initial_guess)):
                parameters[self.parameter_names[i]] = result[0][i]
            self.parameters.append(parameters)
            self.fit.append(__unpack(result))

        return None

    def reset(self):
        """
        Resets the optimization results and fitted parameters.

        Returns:
            None
        """
        self.fit = []
        self.details = []
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
        output = utils.detailed_pandas_compiler(self.fit)
        output.reset_index(drop=True, inplace=True)
        return output
