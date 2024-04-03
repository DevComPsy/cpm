import numpy as np
import pandas as pd
import copy
from . import minimise
from . import utils


def minimum(pars, function, data, loss, prior=True, **args):
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
    llprior = 0
    if prior:
        llprior = function.parameters.prior()
    return metric + llprior


class MCMC:
    """
    Implements an Expectation-Maximisation algorithm for the subject-level parameter estimation.
    """

    def __init__(
        self,
        model=None,
        bounds=None,
        data=None,
        loss=minimise.LogLikelihood.bernoulli,
        chain=4,
        iteration=1000,
        tolerance=1e-6,
        population=100,
        **kwargs
    ):
        self.function = copy.deepcopy(model)
        self.data = data
        self.loss = loss
        self.chain = chain
        self.tolerance = tolerance
        self.iteration = iteration
        self.population = population
        self.kwargs = kwargs
        self.fit = []
        self.details = []

    def run(self):
        for ppt in self.data:

            for i in self.chain:
                tolerate = True
                old = 0
                iteration = 0
                while tolerate:

                    # Generate a proposal distribution and evaluate the likelihood
                    while True:
                        # initialise the parameters
                        proposal = self.function.parameters.sample(
                            size=self.population, jump=True
                        )
                        results = []
                        for pars in proposal:
                            result = minimum(pars, self.function, ppt, self.loss)
                            results.append(result)
                            pass
                        argmax = np.argmax(results)
                        if proposal[argmax] > old:
                            break

                    # Check for convergence
                    if np.abs(proposal[argmax] - old) < self.tolerance:
                        tolerate = False

                    ## update variables
                    old = proposal[argmax]
                    self.fit.append(proposal[argmax], results[argmax])
                    self.function.parameters.update(**proposal[argmax])

                    if not tolerate or iteration == self.iteration:
                        self.maxiter = iteration
                        break

                pass
            pass

    def burnin(self):
        pass
