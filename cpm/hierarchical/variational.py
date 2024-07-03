import numpy as np
import pandas as pd
import copy
from scipy.special import digamma


class VariationalBayes:
    """
    Performs hierarchical Bayesian estimation of a given model using variational (approximate) inference methods.

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
    bounded : array_like, optional
        A vector of 'upper', 'lower', 'both' or 'none' to indicate the bounds of the parameters. If None, assumes that all parameters are bounded. Default is None.

    Notes
    -----

    
    References
    ----------

    Piray, P., Dezfouli, A., Heskes, T., Frank, M. J., & Daw, N. D. (2019). Hierarchical Bayesian inference for concurrent model fitting and comparison for group studies. PLoS computational biology, 15(6), e1007043.
    
    Examples
    --------

    """

    def __init__(
        self,
        optimiser=None,
        objective="minimise",
        iteration=50,
        tolerance_lme=1e-6,
        tolerance_param=0.01,
        chain=4,
        hyperpriors=None, # TODO
        **kwargs,
    ):
        self.function = copy.deepcopy(optimiser.model)
        self.optimiser = copy.deepcopy(optimiser)
        # bounds here should include mean and std for all parameters
        self.iteration = iteration  # maximum number of iterations
        # tolerances for convergence
        self.tolerance_lme = tolerance_lme
        self.tolerance_param = tolerance_param
        self.chain = chain  # number of random parameter initialisations
        self.objective = (
            objective  # whether the optimiser looks for the minimum or maximum
        )
        self.__n_ppt__ = len(self.optimiser.data)
        self.__n_param__ = len(self.optimiser.model.parameters.free())
        self.__n_param_penalty__ = self.__n_param__ * np.log(2 * np.pi)
        self.__param_names__ = self.optimiser.model.parameters.free()
        self.__bounds__ = self.optimiser.model.parameters.bounds()
        # TODO unpack the hyperpriors input argument
        # currently hyperpriors assumed to have entries v, b, s, and a0,
        # following Piray et al., but this is not at all clear
        self.hyperpriors = hyperpriors
        # based on the given parameters of the prior distributions ('hyperpriors')
        # on the population-level means and precisions ('hyperparameters'), we
        # compute some constant variables
        self.__nu__ = self.hyperpriors.v + 0.5 * self.__n_ppt__
        self.__beta__ = self.hyperpriors.b + self.__n_ppt__
        self.__lambda__ = (self.__n_param__ / 2) * (
            digamma(self.__nu__) - np.log(self.__nu__) - (1 / self.__beta__)
        )

        self.kwargs = kwargs

        self.details = []
        self.lmes = []
        self.hyperparameters = pd.DataFrame()
        self.fit = pd.DataFrame()

    # function to update participant-level variables
    def update_participants(self, iter_idx=0, chain_idx=0):

        # delete any pre-existing optimisation output / parameter estimates
        self.optimiser.reset()

        # run the optimisation
        self.optimiser.optimise()

        # extract detailed output of optimisation, and append to self.details                
        details = self.optimiser.fit
        self.details.append(copy.deepcopy(details))

        # extract the parameter estimates and organise them into an array 
        parameters = self.optimiser.parameters
        param = np.zeros(self.__n_ppt__, self.__n_param__)
        for i, name in enumerate(self.__param_names__):
            for ppt, content in enumerate(parameters):
                param[ppt, i] = content.get(name)

        # additionally organise the parameter estimates in a pandas dataframe
        # in long format, to be appended to self.fit
        parameter_long = pd.DataFrame(param, columns=self.__param_names__)
        parameter_long["ppt"] = [i for i in range(self.__n_ppt__)]
        parameter_long["iteration"] = iter_idx + 1
        parameter_long["chain"] = chain_idx
        self.fit = pd.concat([self.fit, parameter_long]).reset_index(drop=True)

        # extract the participant-wise unnormalised log posterior density at the
        # optimised parameter values
        log_posterior = np.asarray([ppt.get("fun") for ppt in details])
        # if the optimiser minimses rather than maximises the target function,
        # the target function is the negative of the log posterior density function.
        # Thus, we multiply by minus 1 to get the log posterior density.
        if self.objective == "minimise":
            log_posterior = -1 * log_posterior

        # extract the Hessian matrix of the target function evaluated at the
        # optimised parameter values
        hessian = []
        for i, ppt in enumerate(details):
            hessian.append(ppt.get("hessian"))

        hessian = np.asarray(hessian)

        # the Hessian matrix is assumed to contain the second derivatives of the
        # negative of the log posterior density function. So, if the objective
        # was to maximise, we need to multiply the entries of the Hessian by -1.
        if self.objective != "minimise":
            hessian = -1 * hessian

        return param, log_posterior, hessian
    
    # function to approximate the participant-wise log model evidence using
    # Laplace's approximation.
    # Input:
    # log_post = participant-wise value of log posterior density function at the
    #   mode (i.e., MAP parameter estimates)
    # hessian = participant-wise Hessian matrix of log posterior density
    # function evaluated at the mode (i.e., MAP parameter estimates)
    def get_lme(self, log_post, hessian):

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
        
        # apply the function defined above to the input hessian matrices
        log_dets = np.asarray(list(map(__log_det_hessian, hessian)))

        # compute the participant-wise log model evidence (lme)
        lme = log_post + 0.5 * (self.__n_param_penalty__ - log_dets) + self.__lambda__

        # compute the summed log model evidence, ignoring NaN or non-finite values
        lme_finite = lme[np.isfinite(lme)]
        lme_sum = np.sum(lme_finite)

        return lme, lme_sum
    
    # function to update population-level parameters based on result of
    # participant-wise optimisation
    def update_population(self, param, hessian, lme, iter_idx=0, chain_idx=0):

        # convenience function to obtain the (pseudo-)inverse of a matrix
        def __inv_mat(x):
            try:
                inv_x = np.linalg.inv(x)
            except np.linalg.LinAlgError:
                inv_x = np.linalg.pinv(x)

            return inv_x

        # get empirical mean of parameter estimates, ignoring NaN or non-finite
        # values
        param[np.isinf(param)] = np.nan
        empirical_means = np.nanmean(param, axis=0)

        # get empirical variance of parameter estimates, ignoring NaN or
        # non-finite values.
        # to this end, we first need to obtain the "within-participant"
        # variances (uncertainties) of the parameter estimates, which are given
        # by the diagonal elements of the matrix inverse of the Hessian
        inv_hessian = np.asarray(
            list(map(__inv_mat, hessian))
        ) # shape: ppt x params x params
        param_uncertainty = np.diagonal(
            inv_hessian, axis1=1, axis2=2
        )  # shape: ppt x params
        param_uncertainty = param_uncertainty.copy()
        # set any non-finite or non-positive values to NaN
        param_uncertainty[
            np.logical_not(np.isfinite(param_uncertainty))
            | (param_uncertainty <= 0)
        ] = np.nan
        # for each parameter, compute the sum across participants of the squared
        # estimate and the uncertainty of the estimate.
        # empirical variance is then computed as the mean of that term (across 
        # participants), minus the square of the empirical means.
        param_var_mat = np.square(param) + param_uncertainty  # shape: ppt x params
        param_var_mat[np.logical_not(np.isfinite(param_var_mat))] = np.nan
        mean_squares = np.nanmean(param_var_mat, axis=0)
        square_means = np.square(empirical_means)
        empirical_variances = mean_squares - square_means
        # also create empirical estimates of standard deviations (square root
        # of empirical variances), ensuring variances are not smaller than 1e-6
        empirical_SDs = np.sqrt(np.clip(empirical_variances, 1e-6, None))

        # compute the expected value (mean) of the posterior distribution of the
        # population-level means (mu) of model parameters
        E_mu = (1 / self.__beta__) * (
            self.__n_ppt__ * empirical_means +
            self.hyperpriors.b * self.hyperpriors.a0
        )

        # compute the expected value (mean) of the posterior distribution of the
        # population-level precisions (tau) of model-parameters
        sq_meandev = np.square(empirical_means - self.hyperpriors.a0)
        scaled_sq_meandev = sq_meandev * (
            (self.hyperpriors.b * self.__n_ppt__) / self.__beta__
        )
        sigma = self.hyperpriors.s + 0.5 * (
            self.__n_ppt__ * empirical_variances + scaled_sq_meandev
        )
        E_tau = sigma / self.__nu__
        # now we need to convert these estimates to usable estimates of standard
        # deviations. to this end, we (1) take the inverse of the estimated
        # precisions to get estimated variances, (2) ensure the estimated
        # variances are not unreasonably small (using 1e-6 as lower threshold),
        # (3) take the square root of the estimated variances to get estimated
        # standard deviations.
        E_sd = np.sqrt(np.clip((1 / E_tau), 1e-6, None))

        # also compute "hierarchical errorbars", which can be used post-hoc for
        # statistical inference on the estimated population-level means
        E_mu_error = np.sqrt(
            (2 * sigma / self.__beta__) / (2 * self.__nu__)
        )

        # use these estimates of the population-level mean and variance to
        # update the priors on model parameters, for next round of
        # participant-wise MAP estimation
        population_updates = {}
        for i, name in enumerate(self.__param_names__):
            population_updates[name] = {
                "mean": E_mu[i],
                "sd": E_sd[i],
            }

        self.optimiser.model.parameters.update_prior(**population_updates)

        # organise population-level variables into a pandas dataframe in long format
        hyper = pd.DataFrame(
            [0, 0, 0, 0, 0, 0],
            index=[
                "chain", "iteration", "parameter",
                "mean", "mean_errorbar", "sd", "lme"
            ],
        ).T
        for i, name in enumerate(self.__param_names__):
            hyper["parameter"] = name
            hyper["mean"] = E_mu[i]
            hyper["mean_errorbar"] = E_mu_error[i]
            hyper["sd"] = E_sd[i]
            hyper["iteration"] = iter_idx + 1
            hyper["chain"] = chain_idx
            hyper["lme"] = lme
            self.hyperparameters = pd.concat([self.hyperparameters, hyper])

        # lastly, calculate the empirical means divided by the empirical SDs,
        # resulting in "normalised" means a.k.a. signal-to-noise ratios, which
        # may be used to track convergence
        param_snr = empirical_means / empirical_SDs

        return population_updates, param_snr
    

    # function to check if the algorithm has converged
    def check_convergence(
            self, lme_new, lme_old, param_snr_new, param_snr_old,
            iter_idx=0, use_lme=True, use_param=True
    ):
        
        print(f"Iteration: {iter_idx + 1}, LME: {lme_new}")

        lme_satisfied = False
        param_satisfied = False
        convergence = False

        if iter_idx > 0:
            if use_lme:
                delta_lme = np.abs(lme_new - lme_old)
                lme_satisfied = delta_lme < self.tolerance_lme
            if use_param:
                param_snr_delta = param_snr_new - param_snr_old
                delta_param = np.sqrt(np.mean(np.square(param_snr_delta)))
                param_satisfied = delta_param < self.tolerance_param

            if lme_satisfied or param_satisfied:
                convergence = True
        
        return convergence
    

    # function to run the hierarchical bayesian inference algorithm
    def run_vb(self, chain_index=0):

        lme_old = np.nan
        lmes = []
        param_snr_old = np.nan

        for iteration in range(self.iteration):

            iter_index = iteration + 1

            # STEP 1: Perform participant-wise optimisation
            param, log_posterior, hessian = self.update_participants(
                iter_idx=iter_index, chain_idx=chain_index
            )

            # STEP 2: Estimate the participant-wise log model evidence
            lme_vector, lme_sum = self.get_lme(
                log_post=log_posterior, hessian=hessian
            ) # TODO add participant-wise lme estimates (`lme_vector`) to output somehow
            lme_sum = copy.deepcopy(lme_sum)
            lmes.append(lme_sum)
            self.lmes.append(lmes)

            # STEP 3: Estimate posterior means of population-level means (mu) and
            # precisions (tau), and use these estimates to update the normal prior
            # on participant-level parameters
            population_updates, param_snr = self.update_population(
                param=param, hessian=hessian, lme = lme_sum,
                iter_idx=iter_index, chain_idx=chain_index
            )

            # STEP 4: Check for convergence based on the change in LME and/or
            # change in standardized estimates of population-level means
            convergence = self.check_convergence(
                lme_new=lme_sum, lme_old=lme_old,
                param_snr_new=param_snr, param_snr_old=param_snr_old,
                iter_idx=iter_index
            )
            if convergence:
                break
            else:
                lme_old = lme_sum
                param_snr_old = param_snr

        # put together a basic summary of results, and return
        output = {
            "lme": lmes,
            "hyperparameters": population_updates,
            "parameters": self.optimiser.model.parameters,
        }

        return output

    def optimise(self):
        """

        """
        output = []
        for chain in range(self.chain):
            print(f"Chain: {chain + 1}")
            results = self.run_vb(chain_index=chain)
            output.append(copy.deepcopy(results))
        self.output = output
        return None
