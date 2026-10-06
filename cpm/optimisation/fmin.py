from ..core.generators import generate_guesses
from ..core.optimisers import (
    objective,
    evaluate_fit,
    fit_extras,
    numerical_hessian,
    group_identifier,
    prepare_data,
)
from ..core.data import detailed_pandas_compiler, decompose
from ..core.parallel import detect_cores, execute_parallel
from ..generators import Simulator, Wrapper

from scipy.optimize import fmin, fmin_l_bfgs_b

import numpy as np
import pandas as pd
import multiprocess as mp
import copy
import inspect
import warnings

__all__ = ["Fmin", "FminBound"]

## SciPy 1.18.0 removed the `disp` and `iprint` options of the L-BFGS-B solver,
## deprecated since 1.15.0, so they can only be forwarded to older SciPy.
LBFGSB_VERBOSITY_SUPPORTED = "disp" in inspect.signature(fmin_l_bfgs_b).parameters


def gradient_by_forward_differences(fun, x, f0, lower, upper, epsilon):
    """
    The gradient of `fun` at `x` by forward differences, as SciPy's L-BFGS-B estimates it with `approx_grad=True`.

    Parameters
    ----------
    fun : callable
        The function, of a 1-D array.
    x : numpy.ndarray
        The point, within the bounds.
    f0 : float
        `fun(x)`.
    lower, upper : numpy.ndarray
        The bounds of each element of `x`, possibly infinite.
    epsilon : float
        The step.

    Returns
    -------
    numpy.ndarray
        The gradient.

    Notes
    -----
    This follows `scipy.optimize._numdiff.approx_derivative` with the "2-point"
    method and an absolute step: a step that would leave the bounds is taken in
    the other direction, and each difference is divided by the step as it is
    represented, ``(x + h) - x``. The gradient is therefore exactly the one SciPy
    computes, so the optimiser takes exactly the same path, but without SciPy's
    overhead on every evaluation. It works on Python floats, which follow the
    same IEEE arithmetic as NumPy's and are faster for a few parameters.
    """
    gradient = np.empty(x.size)
    for i, (value, low, high) in enumerate(zip(x.tolist(), lower.tolist(), upper.tolist())):
        h = epsilon
        ## a step too small to change x falls back to SciPy's default relative step
        if (value + h) - value == 0:
            h = _SQRT_EPS * (1.0 if value >= 0 else -1.0) * max(1.0, abs(value))
        lower_distance, upper_distance = value - low, high - value
        if abs(h) <= max(lower_distance, upper_distance):
            if value + h < low or value + h > high:
                h = -h
        elif upper_distance >= lower_distance:
            h = upper_distance
        else:
            h = -lower_distance
        step = x.copy()
        step[i] = value + h
        gradient[i] = (fun(step) - f0) / ((value + h) - value)
    return gradient


_SQRT_EPS = float(np.finfo(float).eps ** 0.5)


def lbfgsb_options(display=False, **kwargs):
    """
    Assemble the keyword arguments for `scipy.optimize.fmin_l_bfgs_b`, keeping the solver's verbosity options only where the installed SciPy still accepts them.

    Parameters
    ----------
    display : bool
        Whether the solver should report its own progress. Ignored on SciPy 1.18.0 and later.
    **kwargs : dict
        The remaining keyword arguments forwarded to the solver.

    Returns
    -------
    dict
        The keyword arguments to forward to `scipy.optimize.fmin_l_bfgs_b`.

    Notes
    -----
    `disp` is only added when `display` is truthy, so that the default case does not trip the deprecation warning SciPy 1.15.0 to 1.17.x emits for these options.
    """
    ignored = [key for key in ("disp", "iprint") if key in kwargs]
    if display:
        ignored.append("disp")

    if LBFGSB_VERBOSITY_SUPPORTED:
        if display:
            kwargs["disp"] = display
        return kwargs

    for key in ("disp", "iprint"):
        kwargs.pop(key, None)
    if ignored:
        warnings.warn(
            f"The installed SciPy no longer supports the {', '.join(sorted(set(ignored)))} "
            "option(s) of the L-BFGS-B solver, removed in SciPy 1.18.0, so they are ignored. "
            "`display` still reports the progress of the multi-start loop.",
            RuntimeWarning,
            stacklevel=3,
        )
    return kwargs


class Fmin:
    """
    Class representing the Fmin search (unbounded) optimization algorithm using a downhill simplex.

    Parameters
    ----------
    model : cpm.generators.Wrapper
        The model to be optimized.
    data : pd.DataFrame, pd.DataFrameGroupBy, list
        The data used for optimization. If a pd.Dataframe, it is grouped by the `ppt_identifier`. If it is a pd.DataFrameGroupby, groups are assumed to be participants. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    minimisation : function
        The loss function for the objective minimization function. See the `minimise` module for more information. User-defined loss functions are also supported.
    prior: bool
        Whether to include the prior in the optimization. Default is `False`. When `True`, each entry of `fit` additionally records `log_likelihood` and `log_prior`, the summed log likelihood and the summed log prior density at the optimised parameter values, because `fun` is then the negative summed log posterior density and cannot be split apart after the fact.
    metrics : dict, iterable, callable or None
        Goodness-of-fit metrics to evaluate at the optimised parameter values and record alongside the fit, so that they reach `export()` too. Supply a mapping of output names to callables, an iterable of callables named after themselves, or a single callable. Each is called with keyword arguments only - `likelihood` and `log_likelihood` (both the summed log likelihood), `log_prior`, `predicted`, `observed`, `n`, `k` and `parameters` - so a metric takes what it needs and absorbs the rest in `**kwargs`, as `cpm.optimisation.compare.PenalisedLikelihoods` and the loss functions in `cpm.optimisation.minimise` already do. Default is `None`.
    number_of_starts : int
        The number of random initialisations for the optimization. Default is `1`.
    initial_guess : list or array-like
        The initial guess for the optimization. Default is `None`. If `number_of_starts` is set, and the `initial_guess` parameter is 'None', the initial guesses are randomly generated from a uniform distribution.
    parallel : bool
        Whether to use parallel processing. Default is `False`.
    cl : int
        The number of cores to use for parallel processing. Default is `None`. If `None`, the number of cores is set to 2.
        If `cl` is set to `None` and `parallel` is set to `True`, the number of cores is set to the number of cores available on the machine.
    libraries : list, optional
        The libraries to import for parallel processing for `ipyparallel` with the IPython kernel. Default is `["numpy", "pandas"]`.
    ppt_identifier : str
        The column (or the key in the participant data dictionaries) that contains the participant identifier, returned in the optimization details. Required if `data` is a pd.DataFrame. If `data` is a pd.DataFrameGroupBy grouped by a single column, such as `data.groupby("ppt")`, it defaults to the name of that column. Default is `None`.
    **kwargs : dict
        Additional keyword arguments. See the :func:`scipy.optimize.fmin` documentation for what is supported.

    Attributes
    ----------
    number_of_starts : int
        The number of starts requested, retained so that it is recoverable from
        a constructed optimiser rather than only implied by
        `initial_guess.shape[0]`.
    initial_guess_supplied : bool
        Whether `initial_guess` currently holds guesses supplied by the user
        (`True`) or guesses drawn from the parameter bounds (`False`). This
        distinguishes a fit started from a fixed point from one started from
        random restarts, which is otherwise not recoverable after fitting.
        `reset(initial_guess=True)` draws new random guesses and therefore sets
        this attribute to `False`.

    Notes
    -----
    The data parameter must contain all input to the model, including the observed data. The data parameter can be a pandas DataFrame, a pandas DataFrameGroupBy object, or a list of dictionaries. If the data parameter is a pandas DataFrame, it is assumed that the data needs to be grouped by the participant identifier, `ppt_identifier`. If the data parameter is a pandas DataFrameGroupBy object, the groups are assumed to be participants. If the data parameter is a list of dictionaries, each dictionary should contain the data for a single participant, including information about the experiment and the results. The observed data for each participant should be included in the dictionary under the key or column 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable calculated by the model Wrapper.

    The optimization process is repeated `number_of_starts` times, and only the best-fitting output from the best guess is stored.
    """

    def __init__(
        self,
        model=None,
        data=None,
        initial_guess=None,
        minimisation=None,
        cl=None,
        parallel=False,
        libraries=["numpy", "pandas"],
        prior=False,
        metrics=None,
        number_of_starts=1,
        ppt_identifier=None,
        display=False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.ppt_identifier = group_identifier(data, ppt_identifier)
        self.data, self.participants, self.groups, self.__pandas__ = prepare_data(
            data, self.ppt_identifier
        )

        self.loss = minimisation
        self.prior = prior
        self.metrics = metrics
        self.kwargs = kwargs
        self.display = display

        self.fit = []
        self.details = []
        self.parameters = []

        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameters.free()
        if not self.parameter_names:
            raise ValueError(
                "The model does not contain any free parameters. Please check the model parameters."
            )
        if isinstance(model, Simulator):
            raise TypeError(
                "The Fmin algorithm is not compatible with the Simulator object."
            )

        self.number_of_starts = number_of_starts
        self.initial_guess_supplied = initial_guess is not None
        self.initial_guess = generate_guesses(
            bounds=self.model.parameters.bounds(),
            number_of_starts=number_of_starts,
            guesses=initial_guess,
            shape=(number_of_starts, len(self.parameter_names)),
        )

        self.__parallel__ = parallel
        self.__current_guess__ = self.initial_guess[0]
        self.__libraries__ = libraries

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = detect_cores()

    def optimise(self):
        """
        Performs the optimization process.

        Returns
        -------
        None
        """

        def __unpack(x, id=None):
            keys = ["xopt", "fopt", "iter", "funcalls", "warnflag", "hessian"]
            keys += fit_extras(self.prior, self.metrics)
            if id is not None:
                keys.append(id)
            out = {}
            for i in range(len(keys)):
                out[keys[i]] = x[i]
            out["fun"] = out.pop("fopt")
            return out

        def __task(participant, **args):

            participant_dc, observed, ppt = decompose(
                participant=participant,
                pandas=self.__pandas__,
                identifier=self.ppt_identifier,
            )

            model.reset(data=participant_dc)

            result = fmin(
                objective,
                x0=self.__current_guess__,
                args=(model, observed, loss, prior),
                disp=self.display,
                **self.kwargs,
                full_output=True,
            )

            def f(x):
                return objective(x, model, observed, loss, prior)

            hessian = numerical_hessian(func=f, params=result[0] + 1e-3)
            result = (*result, hessian)

            if prior or self.metrics:
                result = (
                    *result,
                    *evaluate_fit(
                        result[0], model, observed, loss, prior, self.metrics
                    ).values(),
                )

            # if participant data contains identifiers, return the identifiers too
            result = (*result, ppt)
            return result

        def __extract_nll(result):
            output = np.zeros(len(result))
            for i in range(len(result)):
                output[i] = result[i][1]
            return output.copy()

        loss = self.loss
        model = self.model
        prior = self.prior

        for i in range(len(self.initial_guess)):
            if self.display:
                print(
                    f"Starting optimization {i+1}/{len(self.initial_guess)} from {self.initial_guess[i]}"
                )
            self.__current_guess__ = self.initial_guess[i]
            if self.__parallel__:
                results = execute_parallel(
                    job=__task,
                    data=self.data,
                    pandas=self.__pandas__,
                    method=None,
                    cl=self.cl,
                    libraries=self.__libraries__,
                )
            else:
                results = list(map(__task, self.data))

            ## extract the negative log likelihoods for each ppt
            if i == 0:
                old_nll = __extract_nll(results)
                self.details = copy.deepcopy(results)
                parameters = {}
                for result in results:
                    for i in range(len(self.parameter_names)):
                        parameters[self.parameter_names[i]] = copy.deepcopy(
                            result[0][i]
                        )
                    self.parameters.append(copy.deepcopy(parameters))
                    self.fit.append(
                        __unpack(copy.deepcopy(result), id=self.ppt_identifier)
                    )
            else:
                nll = __extract_nll(results)
                # check if ppt fit is better than the previous fit
                indices = np.where(nll < old_nll)[0]
                for ppt in indices:
                    self.details[ppt] = copy.deepcopy(results[ppt])
                    for i in range(len(self.parameter_names)):
                        self.parameters[ppt][self.parameter_names[i]] = copy.deepcopy(
                            results[ppt][0][i]
                        )
                    self.fit[ppt] = __unpack(
                        copy.deepcopy(results[ppt]), id=self.ppt_identifier
                    )

        return None

    def reset(self, initial_guess=True):
        """
        Resets the optimization results and fitted parameters.

        Parameters
        ----------
        initial_guess : bool, optional
            Whether to reset the initial guess (generates a new set of random numbers within parameter bounds). Default is `True`.

        Returns
        -------
        None
        """
        self.fit = []
        self.details = []
        self.parameters = []
        if initial_guess:
            self.initial_guess = generate_guesses(
                bounds=self.model.parameters.bounds(),
                number_of_starts=self.number_of_starts,
                guesses=None,
                shape=self.initial_guess.shape,
            )
            # The guesses are now randomly generated, whatever was passed to
            # __init__, so the flag must follow the array it describes.
            self.initial_guess_supplied = False
        return None

    def export(self):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters.
        """
        output = detailed_pandas_compiler(self.fit)
        output.reset_index(drop=True, inplace=True)
        return output


class FminBound:
    """
    Class representing the Fmin search (bounded) optimization algorithm using the L-BFGS-B method.

    Parameters
    ----------
    model : cpm.generators.Wrapper
        The model to be optimized.
    data : pd.DataFrame, pd.DataFrameGroupBy, list
        The data used for optimization. If a pd.Dataframe, it is grouped by the `ppt_identifier`. If it is a pd.DataFrameGroupby, groups are assumed to be participants. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    minimisation : function
        The loss function for the objective minimization function. See the `minimise` module for more information. User-defined loss functions are also supported.
    prior: bool
        Whether to include the prior in the optimization. Default is `False`. When `True`, each entry of `fit` additionally records `log_likelihood` and `log_prior`, the summed log likelihood and the summed log prior density at the optimised parameter values, because `fun` is then the negative summed log posterior density and cannot be split apart after the fact.
    metrics : dict, iterable, callable or None
        Goodness-of-fit metrics to evaluate at the optimised parameter values and record alongside the fit, so that they reach `export()` too. Supply a mapping of output names to callables, an iterable of callables named after themselves, or a single callable. Each is called with keyword arguments only - `likelihood` and `log_likelihood` (both the summed log likelihood), `log_prior`, `predicted`, `observed`, `n`, `k` and `parameters` - so a metric takes what it needs and absorbs the rest in `**kwargs`, as `cpm.optimisation.compare.PenalisedLikelihoods` and the loss functions in `cpm.optimisation.minimise` already do. Default is `None`.
    number_of_starts : int
        The number of random initialisations for the optimization. Default is `1`.
    initial_guess : list or array-like
        The initial guess for the optimization. Default is `None`. If `number_of_starts` is set, and the `initial_guess` parameter is 'None', the initial guesses are randomly generated from a uniform distribution.
    parallel : bool
        Whether to use parallel processing. Default is `False`.
    cl : int
        The number of cores to use for parallel processing. Default is `None`.
        If `cl` is set to `None` and `parallel` is set to `True`, the number of cores is set to the number of cores available on the machine.
    libraries : list, optional
        The libraries to import for parallel processing for `ipyparallel` with the IPython kernel. Default is `["numpy", "pandas"]`.
    ppt_identifier : str
        The column (or the key in the participant data dictionaries) that contains the participant identifier, returned in the optimization details. Required if `data` is a pd.DataFrame. If `data` is a pd.DataFrameGroupBy grouped by a single column, such as `data.groupby("ppt")`, it defaults to the name of that column. Default is `None`.
    **kwargs : dict
        Additional keyword arguments. See the :func:`scipy.optimize.fmin_l_bfgs_b` documentation for what is supported. The solver's own `disp` and `iprint` options, which `display` also sets, were removed in SciPy 1.18.0 and are ignored there.

    Attributes
    ----------
    number_of_starts : int
        The number of starts requested, retained so that it is recoverable from
        a constructed optimiser rather than only implied by
        `initial_guess.shape[0]`.
    initial_guess_supplied : bool
        Whether `initial_guess` currently holds guesses supplied by the user
        (`True`) or guesses drawn from the parameter bounds (`False`). This
        distinguishes a fit started from a fixed point from one started from
        random restarts, which is otherwise not recoverable after fitting.
        `reset(initial_guess=True)` draws new random guesses and therefore sets
        this attribute to `False`.

    Notes
    -----
    The data parameter must contain all input to the model, including the observed data. The data parameter can be a pandas DataFrame, a pandas DataFrameGroupBy object, or a list of dictionaries. If the data parameter is a pandas DataFrame, it is assumed that the data needs to be grouped by the participant identifier, `ppt_identifier`. If the data parameter is a pandas DataFrameGroupBy object, the groups are assumed to be participants. If the data parameter is a list of dictionaries, each dictionary should contain the data for a single participant, including information about the experiment and the results. The observed data for each participant should be included in the dictionary under the key or column 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable calculated by the model Wrapper.

    The optimization process is repeated `number_of_starts` times, and only the best-fitting output from the best guess is stored.
    """

    def __init__(
        self,
        model=None,
        data=None,
        initial_guess=None,
        number_of_starts=1,
        minimisation=None,
        cl=None,
        parallel=False,
        libraries=["numpy", "pandas"],
        prior=False,
        metrics=None,
        ppt_identifier=None,
        display=False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.ppt_identifier = group_identifier(data, ppt_identifier)
        self.data, self.participants, self.groups, self.__pandas__ = prepare_data(
            data, self.ppt_identifier
        )

        self.loss = minimisation
        self.prior = prior
        self.metrics = metrics
        self.kwargs = kwargs
        self.display = display

        self.fit = []
        self.details = []
        self.parameters = []

        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameters.free()
        if not self.parameter_names:
            raise ValueError(
                "The model does not contain any free parameters. Please check the model parameters."
            )
        if isinstance(model, Simulator):
            raise ValueError(
                "The Fmin algorithm is not compatible with the Simulator object."
            )

        self.number_of_starts = number_of_starts
        self.initial_guess_supplied = initial_guess is not None
        self.initial_guess = generate_guesses(
            bounds=self.model.parameters.bounds(),
            number_of_starts=number_of_starts,
            guesses=initial_guess,
            shape=(number_of_starts, len(self.parameter_names)),
        )

        self.__parallel__ = parallel
        self.__current_guess__ = self.initial_guess[0]
        self.__libraries__ = libraries

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = detect_cores()

    def optimise(self, display=True):
        """
        Performs the optimization process.

        Returns
        -------
        None
        """

        def __unpack(x, id=None):
            keys = ["x", "f", "grad", "task", "funcalls", "nit", "warnflag", "hessian"]
            keys += fit_extras(self.prior, self.metrics)
            if id is not None:
                keys.append(id)
            out = {}
            for i in range(len(keys)):
                out[keys[i]] = x[i]
            out["fun"] = out.pop("f")
            return out

        bounds = self.model.parameters.bounds()
        bounds = np.asarray(bounds).T
        bounds = list(map(tuple, bounds))
        loss = self.loss
        model = self.model
        prior = self.prior
        options = lbfgsb_options(display=self.display, **self.kwargs)
        ## with `approx_grad`, the gradient is computed here, exactly as SciPy
        ## would, together with the objective (see `gradient_by_forward_differences`)
        approx_grad = options.pop("approx_grad", False)
        if approx_grad:
            options.pop("fprime", None)
            epsilon = options.pop("epsilon", 1e-8)
            lower, upper = np.asarray(bounds, dtype=float).T
            evaluations_per_point = len(self.parameter_names) + 1
            ## SciPy counts the evaluations of the gradient towards `maxfun`, but
            ## here it only counts the points, so the limit is in points
            options["maxfun"] = options.get("maxfun", 15000) // evaluations_per_point

        def __task(participant, **args):

            participant_dc, observed, ppt = decompose(
                participant=participant,
                pandas=self.__pandas__,
                identifier=self.ppt_identifier,
            )

            model.reset(data=participant_dc)

            if approx_grad:

                def f(x):
                    return objective(x, model, observed, loss, prior)

                def objective_and_gradient(x):
                    f0 = f(x)
                    return f0, gradient_by_forward_differences(f, x, f0, lower, upper, epsilon)

                result = fmin_l_bfgs_b(
                    objective_and_gradient,
                    x0=self.__current_guess__,
                    bounds=bounds,
                    **options,
                )
                result[2]["funcalls"] *= evaluations_per_point
            else:
                result = fmin_l_bfgs_b(
                    objective,
                    x0=self.__current_guess__,
                    bounds=bounds,
                    args=(model, observed, loss, prior),
                    **options,
                )

            def f(x):
                return objective(x, model, observed, loss, prior)

            hessian = numerical_hessian(func=f, params=result[0] + 1e-3)

            result = (*result[0:2], *tuple(list(result[2].values())), hessian)

            if prior or self.metrics:
                result = (
                    *result,
                    *evaluate_fit(
                        result[0], model, observed, loss, prior, self.metrics
                    ).values(),
                )

            result = (*result, ppt)
            return result

        def __extract_nll(result):
            output = np.zeros(len(result))
            for i in range(len(result)):
                output[i] = result[i][1]
            return output.copy()

        for i in range(len(self.initial_guess)):
            if self.display:
                print(
                    f"Starting optimization {i+1}/{len(self.initial_guess)} from {self.initial_guess[i]}"
                )
            self.__current_guess__ = self.initial_guess[i]
            if self.__parallel__:
                results = execute_parallel(
                    job=__task,
                    data=self.data,
                    method=None,
                    cl=self.cl,
                    pandas=self.__pandas__,
                    libraries=self.__libraries__,
                )
            else:
                results = list(map(__task, self.data))

            ## extract the negative log likelihoods for each ppt
            if i == 0:
                old_nll = __extract_nll(results)
                self.details = copy.deepcopy(results)
                parameters = {}
                for result in results:
                    for i in range(len(self.parameter_names)):
                        parameters[self.parameter_names[i]] = copy.deepcopy(
                            result[0][i]
                        )
                    self.parameters.append(copy.deepcopy(parameters))
                    self.fit.append(
                        __unpack(copy.deepcopy(result), id=self.ppt_identifier)
                    )
            else:
                nll = __extract_nll(results)
                # check if ppt fit is better than the previous fit
                indices = np.where(nll < old_nll)[0]
                for ppt in indices:
                    self.details[ppt] = copy.deepcopy(results[ppt])
                    for i in range(len(self.parameter_names)):
                        self.parameters[ppt][self.parameter_names[i]] = copy.deepcopy(
                            results[ppt][0][i]
                        )
                    self.fit[ppt] = __unpack(
                        copy.deepcopy(results[ppt]), id=self.ppt_identifier
                    )

        return None

    def reset(self, initial_guess=True):
        """
        Resets the optimization results and fitted parameters.

        Parameters
        ----------
        initial_guess : bool, optional
            Whether to reset the initial guess (generates a new set of random numbers within parameter bounds). Default is `True`.

        Returns
        -------
        None
        """
        self.fit = []
        self.details = []
        self.parameters = []
        if initial_guess:
            self.initial_guess = generate_guesses(
                bounds=self.model.parameters.bounds(),
                number_of_starts=self.number_of_starts,
                guesses=None,
                shape=self.initial_guess.shape,
            )
            # The guesses are now randomly generated, whatever was passed to
            # __init__, so the flag must follow the array it describes.
            self.initial_guess_supplied = False
        return None

    def export(self):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters.
        """
        output = detailed_pandas_compiler(self.fit)
        output.reset_index(drop=True, inplace=True)
        return output
