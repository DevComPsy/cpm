import numpy as np
import pandas as pd
import copy
from . import minimise
from . import utils


class EmpiricalBayes:
    """
    Implements an Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations.
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
        prior=True,
        **kwargs,
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
        self.prior = prior

    def step(self):
        self.optimiser.run()

    def optimise(self):
        for chain in self.chain:
            # TODO: prior setups
            for iteration in range(self.iteration):
                parameters, hessian, details = self.step()
                self.fit.append(parameters)
                self.details.append(details)

            # TODO: update group-level means and stds
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
