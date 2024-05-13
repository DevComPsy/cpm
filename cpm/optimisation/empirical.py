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


    Notes
    -----
    The EmpiricalBayes class implements an Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations. The algorithm is based on the following steps:

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
        **kwargs,
    ):
        self.function = copy.deepcopy(optimiser.model)
        self.optimiser = copy.deepcopy(optimiser)
        # bounds here should include mean and std for all parameters
        self.iteration = iteration  # maximum number of iterations
        self.tolerance = tolerance  # tolerance for convergence
        self.chain = chain  # number of random parameter initialisations
        self.kwargs = kwargs

        self.__number_of_parameters__ = len(self.optimiser.model.parameters.free())

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
        ## Step numbers correspond to
        lme_old = 0
        lmes = []

        for iteration in range(self.iteration):
            prior = self.optimiser.model.parameters.PDF()

            self.optimiser.reset()

            parameters, hessian, details = self.step()
            self.fit.append(copy.deepcopy(parameters))
            self.details.append(copy.deepcopy(details))

            # DONE: transform negative log likelihood into maximum log likelihood
            negative_log_posterior = np.asarray([-ppt.get("fopt") for ppt in details])
            negative_log_posterior = negative_log_posterior + prior

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
            # TODO: std: make sure the std is not too small by bounding it to 1e-6
            np.clip(between_within_variance, 1e-6, None, out=between_within_variance)
            between_within_variance = between_within_variance.mean(axis=0)
            # TODO: std: fifth subtract the squared mean from result of the above four steps
            variance = between_within_variance.mean(axis=0) - np.square(means)
            # TODO: estimate log model evidence (lme)
            # TODO: lme: first find the log determinant of the hessian matrix for each ppt
            log_determinants = (np.asarray(list(map(np.linalg.slogdet, hessian)))).sum(
                axis=0
            )[1]
            penalty = 0.5 * (
                self.__number_of_parameters__ * np.log(2 * np.pi) - log_determinants
            )
            log_model_evidence = negative_log_posterior + penalty
            # TODO: lme: penalty term - n_params * log(data_points)
            # TODO: lme: third penalise them by the number of parameters (param * log(2 *pi)) - I will skip this step
            # TODO: lme: second sum up the penalised log determinants

            population_updates = {}

            for i, name in enumerate(parameter_names):
                population_updates[name] = {
                    "mean": means[i],
                    "sd": np.sqrt(np.abs(variance[i])),
                }

            self.optimiser.model.parameters.update_prior(**population_updates)

            summed_lme = log_model_evidence.sum()

            lmes.append(copy.deepcopy(summed_lme))

            print(f"Iteration: {iteration + 1}, LME: {summed_lme}")

            if iteration > 0:
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

        # print(f"Optimisation finished in {iteration} iterations: {output}")
        return output

    def optimise(self):
        output = []
        for chain in range(self.chain):
            print(f"Chain: {chain + 1}")
            results = self.stair()
            output.append(copy.deepcopy(results))
        self.output = output
        return output
