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

    """

    def __init__(
        self,
        optimiser=None,
        iteration=1000,
        tolerance=1e-6,
        chain=4,
        prior=True,
        **kwargs,
    ):
        self.function = copy.deepcopy(optimiser.function)
        self.optimiser = copy.deepcopy(optimiser)
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
        hessian = np.zeros(
            (
                len(self.optimiser.details),
                len(self.optimiser.parameter_names),
                len(self.optimiser.parameter_names),
            )
        )
        for i, n in enumerate(self.optimiser.fit):
            hessian[i] = n.get("hessian")
        return self.optimiser.parameters, hessian, self.optimiser.details

    def chain(self):

        log_model_evidence = []
        lme_old = 0

        # TODO: prior setups, calculate group-level means and stds

        population_prior = self.optimiser.model.parameters.PDF(log=True)

        for iteration in range(self.iteration):
            parameters, hessian, details = self.step()
            self.fit.append(copy.deepcopy(parameters))
            self.details.append(copy.deepcopy(details))
            self.optimiser.reset()

            # DONE: transform negative log likelihood into maximum log likelihood
            mll = np.asarray([-ppt.get("fopt") for ppt in self.fit])
            # TODO: HIHG-LEVEL update group-level means and stds

            # TODO: group-level mean is the avarage across participants
            parameter_names = self.optimiser.model.parameters.free()
            param = np.zeros((len(parameters), len(parameter_names)))
            for i, name in enumerate(parameter_names):
                for ppt, content in enumerate(parameters):
                    param[ppt, i] = content.get(name)
            means = param.mean(axis=0)
            # TODO: std: first calculate variance (between-subjects squared differences)
            variance = param.var(axis=0)
            # TODO: std: second calculate inverse hessian matrix
            inv_hessian = np.asarray(list(map(np.linalg.inv, hessian)))

            # TODO: std: third take the diagonal element of the inverse hessian matrix
            diagonal = np.diagonal(inv_hessian, axis1=1, axis2=2)
            # TODO: std: fourth add up variance plus diagonal for each ppt and then add up each ppt and divide them by mean
            between_within_variance = (
                np.square(param - means) + diagonal
            ) / param.shape[0]
            # TODO: std: fifth subtract the squared mean from result of the above four steps
            between_within_variance -= np.square(means)
            # TODO: std: make sure the std is not too small by bounding it to 1e-6
            np.clip(between_within_variance, 1e-6, None, out=between_within_variance)

            # TODO: estimate log model evidence (lme)
            lme = []
            # TODO: lme: first find the log determinant of the hessian matrix for each ppt
            # TODO: lme: second sum up the penalised log determinants and add
            # TODO: lme: third penalise them by the number of parameters (param * log(2 *pi))
            # TODO: lme: sum up ppt lme and add BIC penalty term - n_params * log(data_points)

            population_updates = {}

            for key, value in parameters:
                # NOTE: code is placeholder - dimensions in value will be different
                population_updates.update({key: {"mean": value[0], "std": value[1]}})

            self.optimiser.model.parameters.update_prior(**population_updates)

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
