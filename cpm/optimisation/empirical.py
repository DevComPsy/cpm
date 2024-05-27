import numpy as np
import pandas as pd
import copy


class EmpiricalBayes:
    """
    Implements an Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations.

    Parameters
    ----------
    optimiser : object
        The initialized Optimiser object. It must use an optimisation algorithm that also returns the Hessian matrix.
    objective : str
        The objective of the optimisation, either 'maximise' or 'minimise'. Default is 'minimise'. Only affects how we arrive at the participant-level _a posteriori_ parameter estimates.
    iteration : int, optional
        The maximum number of iterations. Default is 1000.
    tolerance : float, optional
        The tolerance for convergence. Default is 1e-6.
    chain : int, optional
        The number of random parameter initialisations. Default is 4.

    Notes
    -----
    The EmpiricalBayes class implements an Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations. The algorithm is based on the following steps:

    The fitting function must return the Hessian matrix of the optimisation. The Hessian matrix is a square matrix of second-order partial derivatives of the negative log-likelihood function. The Hessian matrix is used to calculate the within-subject variance of the parameters. Compatible methods in the toolbox include Fmin and Fminbound.

    It is also important to note that we will require the Hessian matrix of second derivatives of the negative log posterior (Gershman, 2016, p. 3). This requires us to minimise or maximise the log posterior density as opposed to a simple log likelihood, when estimating participant-level parameters.

    INSERT IMAGE FROM PAPER

    The current implementation also controls for some edge-cases that are not covered by the algorithm above:

    - When calculating the within-subject variance via the Hessian matrix, the algorithm clips the variance to a minimum value of 1e-6 to avoid numerical instability.

    """

    def __init__(
        self,
        optimiser=None,
        iteration=1000,
        tolerance=1e-6,
        chain=4,
        objective="minimise",
        **kwargs,
    ):
        self.function = copy.deepcopy(optimiser.model)
        self.optimiser = copy.deepcopy(optimiser)
        # bounds here should include mean and std for all parameters
        self.iteration = iteration  # maximum number of iterations
        self.tolerance = tolerance  # tolerance for convergence
        self.chain = chain  # number of random parameter initialisations
        self.objective = (
            objective  # whether the optimiser looks for the minimum or maximum
        )
        self.__number_of_parameters__ = len(self.optimiser.model.parameters.free())

        self.kwargs = kwargs

        self.fit = []
        self.details = []
        self.lmes = []

    def step(self):
        self.optimiser.optimise()
        hessian = []
        for i, n in enumerate(self.optimiser.fit):
            hessian.append(n.get("hessian"))

        hessian = np.asarray(hessian)
        return self.optimiser.parameters, hessian, self.optimiser.fit

    def stair(self):
        ## Equation Numbers refer to Equations in the Gershman (2016) Empirical priors for reinforcement learning models
        ## Step numbers correspond to
        lme_old = 0
        lmes = []

        for iteration in range(self.iteration):
            prior = self.optimiser.model.parameters.PDF()

            self.optimiser.reset()

            parameters, hessian, details = self.step()
            self.fit.append(copy.deepcopy(parameters))
            self.details.append(copy.deepcopy(details))

            # extract and transform negative log posteriors into maximum log likelihood
            # this essentially gives us the output of Equation 5
            negative_log_posterior = np.asarray([ppt.get("fun") for ppt in details])
            if self.objective == "minimise":
                negative_log_posterior = -1 * negative_log_posterior

            # get group-level mean across participants
            parameter_names = self.optimiser.model.parameters.free()
            param = np.zeros((len(parameters), len(parameter_names)))
            for i, name in enumerate(parameter_names):
                for ppt, content in enumerate(parameters):
                    param[ppt, i] = content.get(name)

            means = param.mean(axis=0)
            ## In what follows, we break down Equation 6
            # first calculate variance (between-subjects squared differences)
            variance = param.var(axis=0)

            ## getting the diagonal of the inverse hessian matrix
            ## 1. calculate inverse hessian matrix
            print(hessian)
            inv_hessian = np.asarray(list(map(np.linalg.inv, hessian)))
            ## 2. take the diagonal element of the inverse hessian matrix
            diagonal = np.diagonal(inv_hessian, axis1=1, axis2=2)

            ## calculate the variance of the between-subjects squared differences and the within-subjects squared differences
            ## add up variance plus diagonal for each ppt and then add up each ppt and divide them by mean
            between_within_variance = np.square(param - means) + diagonal
            ## get the mean of the between_within_variance minus the
            variance = between_within_variance.mean(axis=0) - np.square(means)
            ## make sure the STD is not too small by bounding it to 1e-6
            np.clip(variance, 1e-6, None, out=variance)

            ## update population-level parameters.
            population_updates = {}

            for i, name in enumerate(parameter_names):
                population_updates[name] = {
                    "mean": means[i],
                    "sd": variance[i],
                }

            self.optimiser.model.parameters.update_prior(**population_updates)

            # estimate log model evidence (lme)
            # first find the log determinant of the hessian matrix for each ppt
            log_determinants = (np.asarray(list(map(np.linalg.slogdet, hessian)))).sum(
                axis=0
            )[1]
            ## add a penalty term to the negative log posterior
            penalty = 0.5 * (
                self.__number_of_parameters__ * np.log(2 * np.pi) - log_determinants
            )
            ## calculate the log model evidence with a penalty and sum them up
            log_model_evidence = negative_log_posterior + penalty
            summed_lme = log_model_evidence.sum()
            ## store the log model evidence
            lmes.append(copy.deepcopy(summed_lme))

            print(f"Iteration: {iteration + 1}, LME: {summed_lme}")

            if iteration > 1:
                if np.abs(summed_lme - lme_old) < self.tolerance:
                    break
                else:  # update the log likelihood
                    lme_old = summed_lme

            iteration += 1

            self.lmes.append(lmes)

        output = {
            "lme": lmes,
            "hyperparameters": population_updates,
            "parameters": self.optimiser.model.parameters,
        }

        # print(f"Chain finished in {iteration} iterations: {population_updates}")
        return output

    def optimise(self):
        output = []
        for chain in range(self.chain):
            print(f"Chain: {chain + 1}")
            results = self.stair()
            output.append(copy.deepcopy(results))
        self.output = output
        return None
