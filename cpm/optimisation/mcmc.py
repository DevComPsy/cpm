import numpy as np
import copy
import pandas as pd
import numpy as np
import copy
from . import minimise
from . import utils
from ..generators import Simulator, Wrapper


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
    observed = copy.deepcopy(data)
    metric = loss(predicted, observed, negative=False, **args)
    del predicted, observed
    if metric == float("inf") or metric == float("-inf") or metric == float("nan"):
        metric = 1e10
    prior = function.parameters.prior()
    return metric + prior


class ExpectationMaximisation:
    """ """

    def __init__(
        self,
        model=None,
        bounds=None,
        data=None,
        loss=minimise.LogLikelihood.bernoulli,
        chain=4,
        iteration=1000,
        tolerance=1e-6,
        **kwargs
    ):
        self.function = copy.deepcopy(model)
        self.data = data
        self.loss = loss
        self.chain = chain
        self.tolerance = tolerance
        self.iteration = iteration
        self.kwargs = kwargs
        self.fit = []
        self.details = []

    def run(self):
        for i in self.chain:
            tolerate = True

            while tolerate:
                # NOTE: placeholders
                proposal = self.function.parameters.sample()
                proposal = np.array(list(proposal))
                result = minimise(
                    minimum, proposal, self.function, self.data, self.loss
                )
                ll = self.function.parameters.prior() + result
            pass
        pass


class EmpiricalBayes:
    """
    Implements an Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model.
    """

    def __init__(
        self,
        optimiser=None,
        parameters=None,
        data=None,
        bounds=None,
        loss=None,
        iteration=1000,
        tolerance=1e-6,
        chain=4,
        **kwargs
    ):
        self.function = copy.deepcopy(optimiser.function)
        self.optimiser = copy.deepcopy(optimiser)
        self.data = data
        self.parameters = parameters
        # bounds here should include mean and std for all parameters
        self.bounds = bounds
        self.loss = loss
        self.iteration = iteration
        self.tolerance = tolerance
        self.chain = chain
        self.kwargs = kwargs
        self.fit = []
        self.details = []
        if hasattr(simulator, "bounds"):
            self.bounds = simulator.bounds
        else:
            self.bounds = bounds

    def optimise(self):
        for i in self.chain:
            # NOTE: palceholders
            self.optimise(
                minimise, self.bounds, self.iteration, self.tolerance, **self.kwargs
            )
            self.fit.append([])
            self.details.append([])
            pass

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
