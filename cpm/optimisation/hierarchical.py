import numpy as np
import pandas as pd
import copy
from . import minimise
from . import utils
from ..generators import Parameters, LogParameters


class EmpiricalBayes:
    """
    Implements an Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations.

    Parameters
    ----------
    optimiser : object
        The initialized Optimiser object. It must use an optimisation algorithm that also returns the Hessian matrix.
    population_parameters : list
        A list of the group-level parameters that will be optimised.

    """

    def __init__(
        self,
        optimiser=None,
        population_parameters=None,
        data=None,
        bounds=None,
        iteration=1000,
        tolerance=1e-6,
        chain=4,
        prior=True,
        **kwargs,
    ):
        self.function = copy.deepcopy(optimiser.function)
        self.optimiser = copy.deepcopy(optimiser)
        self.data = data
        self.pop_parameters = population_parameters
        # bounds here should include mean and std for all parameters
        self.iteration = iteration  # maximum number of iterations
        self.tolerance = tolerance  # tolerance for convergence
        self.chain = chain  # number of random parameter initialisations
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
        hessian = []
        for n in self.optimiser.details:
            hessian.append(n.get("hessian"))
        hessian = np.array(hessian)
        return self.optimiser.parameters, hessian, self.optimiser.details

    def optimise(self):
        for chain in self.chain:
            # TODO: prior setups, calculate group-level means and stds
            for iteration in range(self.iteration):
                parameters, hessian, details = self.step()
                self.fit.append(copy.deepcopy(parameters))
                self.details.append(copy.deepcopy(details))
                self.optimiser.reset()

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
