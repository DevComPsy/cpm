import numpy as np
import pandas as pd
import copy
from ..core.diagnostics import convergence_diagnostics_plots


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
    quiet : bool, optional
        Whether to suppress the output. Default is False.

    Notes
    -----
    The EmpiricalBayes class implements an Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations. For the complete description of the method, please see Gershman (2016).


    The fitting function must return the [Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix) of the optimisation.
    The Hessian matrix is then used in establishing the within-subject variance of the parameters.
    It is also important to note that we will require the Hessian matrix of second derivatives of the **negative log posterior** (Gershman, 2016, p. 3).
    This requires us to minimise or maximise the log posterior density as opposed to a simple log likelihood, when estimating participant-level parameters.

    In the current implementation, we try to calculate the second derivative of the negative log posterior density function according to the following algorithm:

    1. Attempt to use [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition).
    2. If fails, attempt to use [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition).
    3. If fails, attempt to use [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition).
    4. If the result is a complex number with zero imaginary part, keep the real part.

    In addition, because the the Hessian matrix should correspond to the precision matrix, hence its inverse is the variance-covariance matrix, we will use its inverse to calculate the within-subject variance of the parameters. If the algorithm fails to calculate the inverse of the Hessian matrix, it will use the [Moore-Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) instead.

    The current implementation also controls for some **edge-cases** that are not covered by the algorithm above:

    - When calculating the within-subject variance via the Hessian matrix, the algorithm clips the variance to a minimum value of 1e-6 to avoid numerical instability.
    - When calculating the within-subject variance via the Hessian matrix, the algorithm sets any non-finite or non-positive values to NaN.
    - If the second derivative of the negative log posterior density function is not finite, we set the log determinant to -1e6.

    References
    ----------

    Gershman, S. J. (2016). Empirical priors for reinforcement learning models. Journal of Mathematical Psychology, 71, 1-6.

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
        quiet=False,
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
        self.__bounds__ = self.optimiser.model.parameters.bounds()

        self.quiet = quiet

        self.kwargs = kwargs

        self.hyperparameters = pd.DataFrame()
        self.fit = pd.DataFrame()

    def step(self):
        self.optimiser.optimise()
        hessian = []
        for _, n in enumerate(self.optimiser.fit):
            hessian.append(n.get("hessian"))

        hessian = np.asarray(hessian)
        return self.optimiser.parameters, hessian, self.optimiser.fit

    def stair(self, chain_index=0):
        """
        The main function that runs the Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations. This is essentially a single chain.

        Returns
        -------
        dict
            A dictionary containing the log model evidence, the hyperparameters of the group-level distributions, and the parameters of the model.
        """

        # convenience function to obtain the (pseudo-)inverse of a matrix
        def __inv_mat(x):
            try:
                inv_x = np.linalg.inv(x)
            except np.linalg.LinAlgError:
                inv_x = np.linalg.pinv(x)

            return inv_x

        # convenience function to obtain the log determinant of a Hessian matrix
        def __log_det_hessian(x):

            # local convenience function to determine if input is a
            # complex number with non-zero imaginary part
            def has_nonzero_imaginary(x) -> bool:
                if np.iscomplex(x):
                    return np.imag(x) != 0
                return False

            # first attempt using Cholesky decomposition, which is the most efficient
            try:
                L = np.linalg.cholesky(x)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                if has_nonzero_imaginary(log_det):
                    raise np.linalg.LinAlgError
            # second attempt using `slogdet`, which uses LU decomposition
            except np.linalg.LinAlgError:
                try:
                    sign, log_det = np.linalg.slogdet(x)
                    if sign == 0 or has_nonzero_imaginary(log_det):
                        raise np.linalg.LinAlgError
                # third and final attempt using QR decomposition
                except np.linalg.LinAlgError:
                    try:
                        Q, R = np.linalg.qr(x)
                        log_det = np.sum(np.log(np.abs(np.diag(R))))
                        if has_nonzero_imaginary(log_det):
                            return np.nan
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

            self.optimiser.reset()

            # perform participant-wise optimisation, extracting MAP parameter estimates,
            # the Hessian matrix of the target function evaluated at the MAP parameter estimates,
            # and the full output from model fitting
            parameters, hessian, details = self.step()

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

            # organise parameter estimates in an array
            parameter_names = self.optimiser.model.parameters.free()
            param = np.zeros(
                (len(parameters), len(parameter_names))
            )  # shape: ppt x params
            for i, name in enumerate(parameter_names):
                for ppt, content in enumerate(parameters):
                    param[ppt, i] = content.get(name)

            parameter_long = pd.DataFrame(param, columns=parameter_names)
            parameter_long["ppt"] = [i for i in range(len(parameters))]
            parameter_long["iteration"] = iteration + 1
            parameter_long["chain"] = chain_index
            self.fit = pd.concat([self.fit, parameter_long]).reset_index(drop=True)
            # turn any non-finite values into NaN, to avoid subsequent issues
            # with calculating parameter means and variances
            param[np.isinf(param)] = np.nan

            # get estimates of population-level means of parameters
            means = np.nanmean(param, axis=0)  # shape: 1 x params

            # get estimates of population-level variances of parameters.
            # this requires accounting for both the "within-subject" variance (i.e.
            # uncertainty of parameter estimates) and the "between-subject" variance
            # (i.e. individual differences relative to mean)

            # the Hessian matrix should correspond to the precision matrix, hence its
            # inverse is the variance-covariance matrix.
            inv_hessian = np.asarray(
                list(map(__inv_mat, hessian))
            )  # shape: ppt x params x params
            # diagonal elements should correspond to variances (uncertainties)
            param_uncertainty = np.diagonal(
                inv_hessian, axis1=1, axis2=2
            )  # shape: ppt x params
            param_uncertainty = param_uncertainty.copy()  # make sure array is writable
            # keep the signs of the derivatives
            # # set any non-finite or non-positive values to NaN
            param_uncertainty[
                np.logical_not(np.isfinite(param_uncertainty))
                | (param_uncertainty <= 0)
            ] = np.nan

            # for each parameter, compute the sum across participants of the squared estimate and
            # the uncertainty of the estimate.
            # variance is then estimated as the mean of that term (across participants), minus the
            # square of the estimated mean.
            param_var_mat = np.square(param) + param_uncertainty  # shape: ppt x params
            param_var_mat[np.logical_not(np.isfinite(param_var_mat))] = np.nan
            variance = np.nanmean(param_var_mat, axis=0) - np.square(
                means
            )  # shape: 1 x params
            # make sure the variances are non-negative, setting 1e-6 as a lower threshold
            np.clip(variance, 1e-6, None, out=variance)
            # convert variances to standard deviations
            stdev = np.sqrt(variance)

            # update population-level parameters
            population_updates = {}
            for i, name in enumerate(parameter_names):
                population_updates[name] = {
                    "mean": means[i],
                    "sd": stdev[i],
                }
            # use the updated population-level parameters to update the priors on
            # model parameters, for next round of participant-wise MAP estimation
            self.optimiser.model.parameters.update_prior(**population_updates)

            # Equation 6 in Gershman (2016) provides the formula for the log model evidence
            # how to approximate the log model evidence (lme) a.k.a. marginal likelihood:
            # obtain the log determinant of the hessian matrix for each ppt, and incorporate
            # the number of free parameters to define a penalty term
            log_determinants = np.asarray(list(map(__log_det_hessian, hessian)))
            penalty = 0.5 * (
                self.__number_of_parameters__ * np.log(2 * np.pi) - log_determinants
            )
            # calculate the participant-wise log model evidence as the penalised log posterior density,
            # and then sum them up for an overall measure
            log_model_evidence = log_posterior + penalty
            # clip the log model evidence to avoid numerical instability
            log_model_evidence[~np.isfinite(log_model_evidence)] = -1e6
            # sum the log model evidence across participants
            # it is a log-converted version of Equation 8 in Gershman (2016)
            summed_lme = log_model_evidence.sum()
            lmes.append(copy.deepcopy(summed_lme))

            hyper = pd.DataFrame(
                [0, 0, 0, 0, 0, 0, 0],
                index=[
                    "chain",
                    "iteration",
                    "parameter",
                    "mean",
                    "sd",
                    "lme",
                    "reject",
                ],
            ).T

            for i, name in enumerate(parameter_names):
                hyper["parameter"] = name
                hyper["mean"] = means[i]
                hyper["sd"] = stdev[i]
                hyper["iteration"] = iteration + 1
                hyper["chain"] = chain_index
                hyper["lme"] = copy.deepcopy(summed_lme)
                hyper["reject"] = summed_lme < lme_old
                self.hyperparameters = pd.concat([self.hyperparameters, hyper])

            if self.quiet is False:
                print(f"Iteration: {iteration + 1}, LME: {summed_lme}. ")

            if iteration > 1:
                if np.abs(summed_lme - lme_old) < self.tolerance:
                    break
                else:  # update the summed log model evidence
                    lme_old = summed_lme

        output = {
            "lme": lmes,
            "hyperparameters": population_updates,
            "parameters": self.optimiser.model.parameters,
        }

        return output

    def optimise(self):
        """
        This method runs the Expectation-Maximisation algorithm for the optimisation of the group-level distributions of the parameters of a model from subject-level parameter estimations. This is essentially the main function that runs the algorithm for multiple chains.

        """
        # use the updated population-level parameters to update the priors on
        # model parameters, for next round of participant-wise MAP estimation
        self.optimiser.model.parameters.update_prior(**population_updates)
        output = []
        for chain in range(self.chain):
            ## select a random starting point for each chain
            if chain > 0:
                population_updates = {}
                for i, name in enumerate(parameter_names):
                    population_updates[name] = {
                        "mean": np.random.Generator.beta(2, 2) * self.__bounds__[1][i],
                        "sd": np.random.Generator.beta(2, 2)
                        * (self.__bounds__[1][i] / 2),
                    }
                # use the updated population-level parameters to update the priors on
                # model parameters, for next round of participant-wise MAP estimation
                self.optimiser.model.parameters.update_prior(**population_updates)

            if self.quiet is False:
                print(f"Chain: {chain + 1}")
            results = self.stair(chain_index=chain)
            output.append(copy.deepcopy(results))
        self.output = output
        return None

    def parameters(self):
        """
        Returns the estimated individual-level parameters for each iteration and chain.

        Returns
        -------
        pandas.DataFrame
            The estimated individual-level parameters for each iteration and chain.
        """
        return self.fit

    def priors(self):
        """
        Returns the estimated group-level hyperparameters (priors) for each iteration and chain.

        Returns
        -------
        pandas.DataFrame
            The estimated group-level hyperparameters for each iteration and chain.
        """
        return self.hyperparameters

    def diagnostics(self, show=True, save=False, path=None):
        """
        Returns the convergence diagnostics plots for the group-level hyperparameters.

        Parameters
        ----------
        show : bool, optional
            Whether to show the plots. Default is True.
        save : bool, optional
            Whether to save the plots. Default is False.
        path : str, optional
            The path to save the plots. Default is None.

        Notes
        -----
        The convergence diagnostics plots show the convergence of the log model evidence, the means, and the standard deviations of the group-level hyperparameters.
        It also shows the distribution of the means and the standard deviations of the group-level hyperparameters sampled for each chain.
        """
        convergence_diagnostics_plots(
            self.hyperparameters, show=show, save=save, path=path
        )
