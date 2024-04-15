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
    population_parameters : Parameters
        A list of the group-level parameters that will be optimised using the subject-level parameters. It must be a Parameters object.

    """

    def __init__(
        self,
        optimiser=None,
        population_parameters=None,
        iteration=1000,
        tolerance=1e-6,
        chain=4,
        prior=True,
        **kwargs,
    ):
        self.function = copy.deepcopy(optimiser.function)
        self.optimiser = copy.deepcopy(optimiser)
        self.pop_parameters = population_parameters
        # bounds here should include mean and std for all parameters
        self.iteration = iteration  # maximum number of iterations
        self.tolerance = tolerance  # tolerance for convergence
        self.chain = chain  # number of random parameter initialisations
        self.kwargs = kwargs
        self.fit = []
        self.details = []
        self.prior = prior

        if self.optimiser.prior is not True:
            raise ValueError("The optimiser must be set to use priors.")

    def step(self):
        self.optimiser.run()
        hessian = []
        for n in self.optimiser.details:
            hessian.append(n.get("hessian"))
        hessian = np.array(hessian)
        return self.optimiser.parameters, hessian, self.optimiser.details

    def chain(self):

        log_model_evidence = []
        lme_old = 0

        for iteration in range(self.iteration):
            # TODO: prior setups, calculate group-level means and stds

            population_prior = self.optimiser.model.parameters.PDF(log=True)

            for iteration in range(self.iteration):
                parameters, hessian, details = self.step()
                self.fit.append(copy.deepcopy(parameters))
                self.details.append(copy.deepcopy(details))
                self.optimiser.reset()

            # TODO: transform negative log likelihood into maximum log likelihood
            # TODO: update group-level means and stds
            # TODO: group-level mean is the avarage across participants
            # TODO: std: first calculate variance (between-subjects squared differences)
            # TODO: std: second calculate inverse hessian matrix
            # TODO: std: third take the diagonal element of the inverse hessian matrix
            # TODO: std: fourth add up variance plus diagonal for each ppt and then add up each ppt and divide them by mean
            # TODO: std: fifth subtract the squared mean from result of the above four steps
            # TODO: std: make sure the std is not too small by bounding it to 1e-6

            # TODO: estimate log model evidence (lme)
            lme = []
            # TODO: lme: first find the log determinant of the hessian matrix for each ppt
            # TODO: lme: second sum up the penalised log determinants and add
            # TODO: lme: third penalise them by the number of parameters (param * log(2 *pi))
            # TODO: lme: sum up ppt lme and add BIC penalty term - n_params * log(data_points)

            if iteration > 0:
                if np.abs(lme - lme_old) < self.tolerance:
                    break
                else:  # update the log likelihood
                    lme_old = lme
                    log_model_evidence.append(copy.deepcopy(lme))

            pass

        def optimise(self):
            for chain in self.chain:
                self.chain()

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
