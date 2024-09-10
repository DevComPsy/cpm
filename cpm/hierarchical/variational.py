import numpy as np
import pandas as pd
import copy
import warnings

from scipy.special import digamma
from scipy.stats import t as students_t

from ..generators import Parameters
from ..core.diagnostics import convergence_diagnostics_plots


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
    tolerance_lme : float, optional
        The tolerance for convergence with respect to the log model evidence. Default is 1e-3.
    tolerance_param : float, optional
        The tolerance for convergence with respect to the "normalized" means of parameters. Default is 1e-3.
    chain : int, optional
        The number of random parameter initialisations. Default is 4.
    hyperpriors: dict, optional
        A dictionary of given parameter values of the prior distributions on the population-level parameters (means mu and precisions tau).
    convergence : str, optional
        The convergence criterion. Default is 'parameters'. Options are 'lme' and 'parameters'.

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
        tolerance_lme=1e-3,
        tolerance_param=1e-3,
        chain=4,
        hyperpriors=None,
        convergence="parameters",
        quiet=False,
        **kwargs,
    ):
        # write input arguments to self
        self.optimiser = copy.deepcopy(optimiser)
        self.objective = objective
        self.iteration = iteration
        self.tolerance_lme = tolerance_lme
        self.tolerance_param = tolerance_param
        self.chain = chain
        self.__use_params__ = convergence == "parameters"
        self.__n_param__ = len(self.optimiser.model.parameters.free())
        if hyperpriors is None:
            # default parameter values of prior distributions ('hyperpriors')
            # on population-level means and precisions ('hyperparameters').
            # All of the following default values follow Piray et al. (2019)
            # page 31, section "Parameters, initialization and convergence
            # criteria".
            warnings.warn("Using default hyperpriors.")
            self.hyperpriors = Parameters(
                # vector of means of the normal prior on the population-level
                # means, mu.
                a0=np.zeros(self.__n_param__),
                # scalar value that is multiplied with population-level
                # precisions, tau, to determine the standard deviations of the
                # normal prior on the population-level means, mu.
                b=1,
                # scalar value that is used to determine the shape parameter
                # (nu) of the gamma prior on population-level precisions, tau.
                v=0.5,
                # vector of values that serve as lower bounds on the scale
                # parameters (sigma) of the gamma prior on population-level
                # precisions, tau.
                s=np.repeat(0.01, self.__n_param__),
            )
        else:
            self.hyperpriors = hyperpriors

        self.__quiet__ = quiet
        self.kwargs = kwargs

        # write some further objects to self:
        # copy the given model function
        self.function = copy.deepcopy(optimiser.model)
        # store some useful variables
        self.__n_ppt__ = len(self.optimiser.data)
        self.__param_names__ = self.optimiser.model.parameters.free()
        self.__bounds__ = self.optimiser.model.parameters.bounds()

        # Compute some constant values that will be needed for parameter estimation.
        # Notation follows Piray et al. (2019), dropping subscript k (model indicator).
        # Equation references mirror those in Piray et al. (2019):
        # Equation 22, page 29 (term `D_{k} * log 2pi`)
        self.__n_param_penalty__ = self.__n_param__ * np.log(2 * np.pi)
        # Equation 15, page 28.
        self.__beta__ = self.hyperpriors.b + self.__n_ppt__
        # Equation 17, page 28.
        self.__nu__ = self.hyperpriors.v + 0.5 * self.__n_ppt__
        # Equation 23, page 29.
        self.__lambda__ = (self.__n_param__ / 2) * (
            digamma(self.__nu__) - np.log(self.__nu__) - (1 / self.__beta__)
        )

        # initialise some output objects to self
        self.details = []
        self.lmes = []

        self.hyperparameters = pd.DataFrame()
        self.mu_stats = pd.DataFrame()
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
        param = np.zeros((self.__n_ppt__, self.__n_param__))
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

    def get_lme(self, log_post, hessian):
        """
        Function to approximate the participant-wise log model evidence using Laplace's approximation.

        Parameters
        ----------
        log_post : array-like
            Participant-wise value of log posterior density function at the mode (i.e., MAP parameter estimates).
        hessian : array-like
            Participant-wise Hessian matrix of log posterior density function evaluated at the mode (i.e., MAP parameter estimates).

        Returns
        -------
        lme : array
            Participant-wise log model evidence.
        lme_sum : float
            Summed log model evidence.
        """

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
        # Following line corresponds to Piray et al. (2019) Equation 22, page 29:
        # - log_post corresponds to `log f_{kn}`
        # - self.__n_param_penalty__ corrsponds to `D_{k} log 2pi`
        # - log_dets corresponds to `log |A_{kn}|`
        # - self.__lambda__ corresponds `lambda_{k}`
        # `E[log m_{k}]` is assumed to be equal to log(1) = 0, hence dropped.
        lme = log_post + 0.5 * (self.__n_param_penalty__ - log_dets) + self.__lambda__

        # compute the summed log model evidence, ignoring NaN or non-finite values
        lme_finite = lme[np.isfinite(lme)]
        lme_sum = np.sum(lme_finite)

        return lme, lme_sum

    # function to update population-level parameters based on result of
    # participant-wise optimisation
    def update_population(self, param, hessian, lme, iter_idx=0, chain_idx=0):
        """
        Function to update the population-level parameters based on the results of participant-wise optimisation.

        Parameters
        ----------
        param : array-like
            Participant-wise parameter estimates.
        hessian : array-like
            Participant-wise Hessian matrices of the log posterior density function evaluated at the mode (i.e., MAP parameter estimates).
        lme : float
            Summed log model evidence.
        iter_idx : int, optional
            The iteration index. Default is 0.
        chain_idx : int, optional
            The chain index. Default is 0.

        Returns
        -------
        population_updates : dict
            Dictionary of updated population-level parameters.
        param_snr : array-like
            Standardised estimates of population-level means.
        """

        # convenience function to obtain the (pseudo-)inverse of a matrix
        def __inv_mat(x):
            try:
                inv_x = np.linalg.inv(x)
            except np.linalg.LinAlgError:
                inv_x = np.linalg.pinv(x)

            return inv_x

        # Equation and page numbers refer to Piray et al. (2019).
        # Compared to Piray et al. (2019), we use the following simplifying
        # assumptions:
        # - `r_{kn}` is assumed to be equal to 1 for all n (and k), hence can be
        #   dropped from all equations
        # - `bar{N}_{k}` is assumed to be equal to N

        # get empirical mean of parameter estimates, ignoring NaN or non-finite
        # values
        # The following code corresponds to Equation 12, page 28, where
        # param corresponds to `lambda_{kn}`, hence `bar{lambda}_{k}` is simply
        # the arithmetic mean (given our simplifying assumptions above)
        param[np.isinf(param)] = np.nan
        empirical_means = np.nanmean(param, axis=0)

        # get empirical variance of parameter estimates, ignoring NaN or
        # non-finite values.
        # Following lines correspond to Equation 13, page 28, where
        # - np.square(param) corresponds to `lambda_{kn} lambda_{kn}^{T}`
        # - np.square(empirical_means) corresponds to
        #   `bar_{lambda}_{k} bar_{lambda}_{k}^{T}`
        # - param_uncertainty corresponds to `A_{kn}^{-1}`
        # Confirmed our implementation is consistent with Piray et al.'s code implementation:
        # https://github.com/payampiray/cbm/blob/11b5ad6dbcb0475b49564f8888515a6c06a76f18/codes/cbm_hbi_hbi.m#L258
        #
        # we first need to obtain the "within-participant" variances (uncertainties)
        # of the parameter estimates, which are given by the diagonal elements of
        # the matrix inverse of the Hessian
        inv_hessian = np.asarray(
            list(map(__inv_mat, hessian))
        )  # shape: ppt x params x params
        param_uncertainty = np.diagonal(
            inv_hessian, axis1=1, axis2=2
        )  # shape: ppt x params
        param_uncertainty = param_uncertainty.copy()
        # set any non-finite or non-positive values to NaN
        param_uncertainty[
            np.logical_not(np.isfinite(param_uncertainty)) | (param_uncertainty <= 0)
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
        empirical_SDs = np.sqrt(np.clip(empirical_variances, a_min=1e-6, a_max=None))

        # TODO: ensure that shapes of `empirical_means` and `self.hyperpriors.a0`
        # are consistent

        # compute the expected value (mean) of the posterior distribution of the
        # population-level means (mu) of model parameters
        # Following lines correspond to Equation 14, page 28
        E_mu = (1 / self.__beta__) * (
            self.__n_ppt__ * empirical_means + self.hyperpriors.b * self.hyperpriors.a0
        )

        # compute the expected value (mean) of the posterior distribution of the
        # population-level precisions (tau) of model-parameters
        # Following lines correspond to Equation 16, page 28, where:
        # - sq_meandev corresponds to
        #   `(bar{lambda}_{k} - a_{0})(bar{lambda}_{k} - a_{0})^{T}`
        # - empirical_variances corresponds to `bar{V}_{k}`
        sq_meandev = np.square(empirical_means - self.hyperpriors.a0)
        scaled_sq_meandev = sq_meandev * (
            (self.hyperpriors.b * self.__n_ppt__) / self.__beta__
        )
        sigma = self.hyperpriors.s + 0.5 * (
            self.__n_ppt__ * empirical_variances + scaled_sq_meandev
        )
        # Following line is given in text in between Equations 19 and 20, page 29
        E_tau = sigma / self.__nu__

        # now we need to convert these estimates of population-level precisions
        # into usable estimates of standard deviations.
        # to this end, we (1) take the inverse of the estimated precisions to
        # get estimated variances, (2) ensure the estimated variances are not
        # unreasonably small (using 1e-6 as lower threshold), (3) take the
        # square root of the estimated variances to get estimated SDs.
        E_sd = np.sqrt(np.clip((1 / E_tau), a_min=1e-6, a_max=None))

        # also compute "hierarchical errorbars" - basically standard errors of
        # the estimates of the population-level means, which can be used
        # post-hoc for statistical inference
        # Following line is given in text around Equation 24, page 30.
        # See also Piray et al's code implementation:
        # https://github.com/payampiray/cbm/blob/11b5ad6dbcb0475b49564f8888515a6c06a76f18/codes/cbm_hbi_hbi.m#L168-L174
        E_mu_error = np.sqrt((2 * sigma / self.__beta__) / (2 * self.__nu__))

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
            [0, 0, 0, 0, 0, 0, 0],
            index=[
                "chain",
                "iteration",
                "parameter",
                "mean",
                "mean_errorbar",
                "sd",
                "lme",
            ],
        ).T
        for i, name in enumerate(self.__param_names__):
            hyper["parameter"] = name
            hyper["mean"] = E_mu[i]
            hyper["mean_se"] = E_mu_error[i]
            hyper["sd"] = E_sd[i]
            hyper["iteration"] = iter_idx + 1
            hyper["chain"] = chain_idx
            hyper["lme"] = lme
            self.hyperparameters = pd.concat([self.hyperparameters, hyper])

        # lastly, calculate the empirical means divided by the empirical SDs,
        # resulting in "normalised" means a.k.a. signal-to-noise ratios, which
        # can be used assess convergence
        param_snr = empirical_means / empirical_SDs

        return population_updates, param_snr

    # function to check if the algorithm has converged
    def check_convergence(
        self,
        lme_new,
        lme_old,
        param_snr_new,
        param_snr_old,
        iter_idx=0,
        use_lme=True,
        use_param=True,
    ):
        """
        Function to check if the algorithm has converged.

        Parameters
        ----------
        lme_new : float
            The new log model evidence.
        lme_old : float
            The old log model evidence.
        param_snr_new : array-like
            The new standardised estimates of population-level means.
        param_snr_old : array-like
            The old standardised estimates of population-level means.
        iter_idx : int, optional
            The iteration index. Default is 0.
        use_lme : bool, optional
            Whether to use the log model evidence for checking convergence. Default is True.
        use_param : bool, optional
            Whether to use the standardised estimates of population-level means for checking convergence. Default is True.

        Returns
        -------
        convergence : bool
            Whether the algorithm has converged.
        """

        if self.__quiet__ is False:
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
        """
        Run the hierarchical Bayesian inference algorithm.

        Parameters
        ----------
        chain_index : int, optional
            The chain index. Default is 0.

        Returns
        -------
        output : dict
            Dictionary of results.
        """

        lmes = []
        lme_old = np.nan
        param_snr_old = np.nan

        for iteration in range(self.iteration):

            iter_index = iteration + 1

            # STEP 1: Perform participant-wise optimisation
            param, log_posterior, hessian = self.update_participants(
                iter_idx=iter_index, chain_idx=chain_index
            )

            # STEP 2: Estimate the participant-wise log model evidence
            # TODO use participant-wise lme estimates (`lme_vector`) in output
            lme_vector, lme_sum = self.get_lme(log_post=log_posterior, hessian=hessian)
            lme_sum = copy.deepcopy(lme_sum)
            lmes.append(lme_sum)
            self.lmes.append(lmes)

            # STEP 3: Estimate posterior means of population-level means (mu) and
            # precisions (tau), and use these estimates to update the normal prior
            # on participant-level parameters
            population_updates, param_snr = self.update_population(
                param=param,
                hessian=hessian,
                lme=lme_sum,
                iter_idx=iter_index,
                chain_idx=chain_index,
            )

            # STEP 4: Check for convergence based on the change in LME and/or
            # change in standardized estimates of population-level means
            convergence = self.check_convergence(
                lme_new=lme_sum,
                lme_old=lme_old,
                param_snr_new=param_snr,
                param_snr_old=param_snr_old,
                iter_idx=iter_index,
                use_param=True,
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

    # function to run the VB algorithm for multiple repeats ("chains")
    def optimise(self):
        """
        Run the Variational Bayes algorithm for multiple chains.


        """
        output = []
        for chain in range(self.chain):
            if self.__quiet__ is False:
                print(f"Chain: {chain + 1}")
            results = self.run_vb(chain_index=chain)
            output.append(copy.deepcopy(results))
        self.output = output
        return None

    # function to perform one-sample Student's t-tests on the estimated values
    # of population-level means with respect to given null hypothesis values.
    def ttest(self, null=None):
        """
        Perform a one-sample Student's t-test on the estimated values of population-level means with respect to given null hypothesis values.

        Parameters
        ----------
        null : dict or pd.DataFrame
            The null hypothesis values for the population-level means for each parameters.

        Returns
        -------
        t_df : pd.DataFrame
            The results of the t-test.
        """

        # extract the pandas dataframe containing results regarding the
        # estimated population-level parameters
        hyper_df = self.hyperparameters

        # convert null to pandas dataframe if it is a dictionary
        if isinstance(null, dict):
            null_pd = pd.DataFrame(list(null.items()), columns=["parameter", "null"])
        elif isinstance(null, pd.DataFrame):
            null_df = null_pd
            null_df.columns = ["parameter", "null"]
        elif null is None:
            raise ValueError("The null hypothesis must be provided.")
        else:
            raise ValueError("null should be either a dictionary or pandas DataFrame")

        # merge the given null values with the results dataframe.
        # NB doing inner join, so any parameters not given in null will be
        # excluded from subsequent analysis.
        t_df = hyper_df.merge(null_df, on="parameter", how="inner")

        # calculate the t-statistics
        t_df["t_stat"] = (t_df["mean"] - t_df["null"]) / t_df["mean_se"]
        # calculate the p-value. we do this in two steps:
        # first, calculate the left tail probability: that is, the probability
        # under the null hypothesis that a t-statistic is less than or equal to
        # the negative absolute values of our observed t-statistics.
        # second, we multiply these left tail probabilities by 2, to account for
        # both the left and right tails of the Student's t-distribution (this
        # works because it is a symmetric distribution).
        t_df["p_val"] = 2 * students_t.cdf(
            x=(-1 * np.abs(t_df["t_stat"])), df=(2 * self.__nu__)
        )

        # TODO: check if this is a sensible approach, or if we get issues with
        # overwriting an existing dataframe
        self.mu_stats = pd.concat([self.mu_stats, t_df])

        return t_df

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
        convergence_diagnostics(self.hyperparameters, show=show, save=save, path=path)
