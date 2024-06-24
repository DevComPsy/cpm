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

    The current implementation also controls for some **edge-cases** that are not covered by the algorithm above:

    - When calculating the within-subject variance via the Hessian matrix, the algorithm clips the variance to a minimum value of 1e-6 to avoid numerical instability.

    Examples
    --------
    >>> from cpm.optimisation import EmpiricalBayes
    >>> from cpm.models import DeltaRule
    >>> from cpm.optimisation import FminBound, minimise
    >>>
    >>> model = DeltaRule()
    >>> optimiser = FminBound(
        model=model,
        data=data,
        initial_guess=None,
        number_of_starts=2,
        minimisation=minimise.LogLikelihood.bernoulli,
        parallel=False,
        prior=True,
        ppt_identifier="ppt",
        display=False,
        maxiter=200,
        approx_grad=True
        )
    >>> eb = EmpiricalBayes(optimiser=optimiser, iteration=1000, tolerance=1e-6, chain=4)
    >>> eb.optimise()

    """

    def __init__(
        self,
        optimiser=None,
        objective="minimise",
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
        """
        The main function that runs the Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations. This is essentially a single chain.

        Returns
        -------
        dict
            A dictionary containing the log model evidence, the hyperparameters of the group-level distributions, and the parameters of the model.
        """

        # convenience function to obtain the log determinant of a Hessian matrix
        def __log_det_hessian(x):
            # first attempt using Cholesky decomposition, which is the most efficient
            try:
                L = np.linalg.cholesky(x)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                # force error if solution is complex number with non-zero imaginary part
                if np.iscomplex(log_det) and np.imag(log_det) != 0:
                    raise np.linalg.LinAlgError
            # second attempt using `slogdet`, which uses LU decomposition
            except np.linalg.LinAlgError:
                try:
                    sign, log_det = np.linalg.slogdet(x)
                    # force error if solution is zero or complex number with non-zero imaginary part 
                    if sign == 0 or (np.iscomplex(log_det) and np.imag(log_det) != 0):
                        raise np.linalg.LinAlgError
                # third and final attempt using QR decomposition
                except np.linalg.LinAlgError:
                    try:
                        Q, R = np.linalg.qr(x)
                        log_det = np.sum(np.log(np.abs(np.diag(R))))
                        # give up (NaN) if solution is complex number with non-zero imaginary part
                        if np.iscomplex(log_det) and np.imag(log_det) != 0:
                            return np.nan
                    # give up (NaN) if all matrix decomposition methods failed
                    except np.linalg.LinAlgError:
                        return np.nan
            
            # if solution is complex number with zero imaginary part, just keep the real part
            if np.iscomplex(log_det):
                log_det = np.real(log_det)

            return log_det
        

        # Equation numbers refer to equations in the Gershman (2016) Empirical priors for reinforcement learning models
        lme_old = 0
        lmes = []

        for iteration in range(self.iteration):
            prior = self.optimiser.model.parameters.PDF()

            self.optimiser.reset()

            # perform participant-wise optimisation, extracting MAP parameter estimates,
            # the Hessian matrix of the target function evaluated at the MAP parameter estimates,
            # and the full output from model fitting
            parameters, hessian, details = self.step()
            self.fit.append(copy.deepcopy(parameters))
            self.details.append(copy.deepcopy(details))

            # extract the participant-wise unnormalised log posterior density
            log_posterior = np.asarray([ppt.get("fun") for ppt in details])
            # if the optimiser minimses rather than maximises the target function, then the
            # target function is the _negative_ of the log posterior density function. Thus, we
            # multiply by minus 1 to get the log posterior density.
            if self.objective == "minimise":
                log_posterior = -1 * log_posterior
            # for the Laplace approximation, the Hessian matrix is assumed to contain the second
            # derivatives of the _negative_ log posterior density function. So, if the objective
            # was to maximise, then we need to multiply the entries of the Hessian matrix by -1.
            if self.objective != "minimise":
                hessian = -1 * hessian

            # organise parameter estimates in a participant x parameter array
            parameter_names = self.optimiser.model.parameters.free()
            param = np.zeros((len(parameters), len(parameter_names)))
            for i, name in enumerate(parameter_names):
                for ppt, content in enumerate(parameters):
                    param[ppt, i] = content.get(name)

            # define a mask that excludes any NaN or inf param values
            param_valid = np.isfinite(param)

            # get estimates of population-level means of parameters
            means = np.mean(
                param[param_valid].reshape(param.shape[0], -1), axis=0
            )            

            # get estimates of population-level variances of parameters.
            # this requires accounting for both the "within-subject" variance (i.e.
            # uncertainty of parameter estimates) and the "between-subject" variance
            # (i.e. individual differences relative to mean)

            # 1. "within-subject" variance
            # the Hessian matrix should correspond to the precision matrix, hence its
            # inverse is the variance-covariance matrix.
            try:
                inv_hessian = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                inv_hessian = np.linalg.pinv(hessian)
            
            within_variance = np.diagonal(inv_hessian, axis1=1, axis2=2)
            # the diagonal elements should correspond to variances, so exclude (NaN)
            # any negative or non-finite values
            within_variance[np.logical_not(np.isfinite(within_variance)) |
                            (within_variance < 0)] = np.nan
            
            # 2. "between-subject" variance
            # calculate squared differences from mean, add up the "within-subject" variance,
            # take the mean of that term, and subtract the squared mean
            # TODO account for `param_valid`
            between_within_variance = np.square(param - means) + within_variance
            variance = between_within_variance.mean(axis=0) - np.square(means)
            # make sure the STD is not too small by bounding it to 1e-6
            np.clip(variance, 1e-6, None, out=variance)

            ## update population-level parameters.
            population_updates = {}

            for i, name in enumerate(parameter_names):
                population_updates[name] = {
                    "mean": means[i],
                    "sd": variance[i],
                }

            # use the updated population-level parameters to update the priors on
            # model parameters, for next round of participant-wise MAP estimation
            self.optimiser.model.parameters.update_prior(**population_updates)

            # approximate the log model evidence (lme) a.k.a. marginal likelihood:
            # obtain the log determinant of the hessian matrix for each ppt, and incorporate
            # the number of free parameters to define a penalty term
            log_determinants = np.asarray(list(map(__log_det_hessian, hessian)))
            penalty = 0.5 * (
                self.__number_of_parameters__ * np.log(2 * np.pi) - log_determinants
            )
            # calculate the participant-wise log model evidence with a penalty,
            # and then sum them up for an overall measure
            log_model_evidence = log_posterior + penalty
            summed_lme = log_model_evidence.sum()
            # store the log model evidence
            lmes.append(copy.deepcopy(summed_lme))

            print(f"Iteration: {iteration + 1}, LME: {summed_lme}")

            if iteration > 1:
                if np.abs(summed_lme - lme_old) < self.tolerance:
                    break
                else:  # update the log model evidence
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
        """
        This method runs the Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations. This is essentially the main function that runs the algorithm for multiple chains.

        """
        output = []
        # TODO check on number of cpus
        for chain in range(self.chain):
            print(f"Chain: {chain + 1}")
            results = self.stair()
            output.append(copy.deepcopy(results))
        self.output = output
        return None
