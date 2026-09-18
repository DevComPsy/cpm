import numpy as np
import pandas as pd
import numdifftools as nd
import copy

__all__ = [
    "objective",
    "decompose_objective",
    "normalise_metrics",
    "fit_extras",
    "evaluate_fit",
    "prepare_data",
    "numerical_hessian",
]


def numerical_hessian(func=None, params=None, hessian=None):
    """Calculate numerically the hessian matrix of func with respect to ``params``.

    Args:
        func: Function without arguments that depends on ``params``
        params: Parameters that ``func`` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Hessian matrix
    """

    hesse_func = nd.Hessian(func, step=1e-4, method="forward")
    computed_hessian = hesse_func(params)
    return computed_hessian


def objective(pars, function, data, loss, prior=False):
    """
    The `objective` function calculates a metric by comparing predicted values with
    observed values.

    Parameters
    ----------
    pars
        The `pars` parameter is a dictionary that contains the parameters for the
        function that needs to be minimized.
    function
        The `function` is a cpm.generators.Wrapper or similar class.
    data
        The `data` parameter is the data that is used to compare the predicted values
        with the observed values.
    loss
        The `loss` parameter is the loss function that is used to calculate the metric
        value.
    args
        The `args` parameter is a dictionary that contains additional parameters that
        are used in the loss function.

    Returns
    -------
        The metric value is being returned.

    """
    function.reset(parameters=pars)
    function.run()
    predicted = copy.deepcopy(function.dependent)
    observed = copy.deepcopy(data)
    metric = loss(predicted=predicted, observed=observed)
    del predicted, observed
    if np.isnan(metric) or np.isinf(metric):
        metric = 1e10
    if prior:
        prior_pars = function.parameters.PDF(log=True)
        metric += -prior_pars
    return metric


def decompose_objective(pars, function, data, loss, prior=False):
    """
    Split the value `objective` returns at `pars` into the summed log likelihood and the summed log prior density.

    With `prior=True` the value `objective` minimises is the negative summed log
    posterior density, which on its own cannot be turned back into a log
    likelihood. Model comparison and goodness-of-fit indices generally need the
    two parts separately, so this recomputes them at a given set of parameter
    values.

    Parameters
    ----------
    pars : array-like
        The parameter values to evaluate, normally the optimised ones.
    function : cpm.generators.Wrapper
        The model, as passed to `objective`.
    data
        The observed data, as passed to `objective`.
    loss : callable
        The loss function, as passed to `objective`. See the `minimise` module.
    prior : bool
        Whether the fit included the prior. When `False` the log prior density is
        `0.0`, because no prior entered the objective.

    Returns
    -------
    tuple of float
        The summed log likelihood and the summed log prior density at `pars`.

    Notes
    -----
    This mirrors `objective` step for step, including its substitution of `1e10`
    for a non-finite loss, so that

    ``objective(pars, ...) == -(log_likelihood + log_prior)``

    holds exactly for the same arguments, and the two parts always add back up to
    the `fun` an optimiser reports.
    """
    log_likelihood, log_prior, _, _ = _evaluate(pars, function, data, loss, prior)
    return log_likelihood, log_prior


def _evaluate(pars, function, data, loss, prior=False):
    """
    Run `function` at `pars` and return the pieces `objective` is assembled from.

    Kept separate so that `decompose_objective` and `evaluate_fit` agree by
    construction and cost a single model run between them.

    Returns
    -------
    tuple
        The summed log likelihood, the summed log prior density, the predicted
        values and the observed values.
    """
    function.reset(parameters=pars)
    function.run()
    predicted = copy.deepcopy(function.dependent)
    observed = copy.deepcopy(data)
    metric = loss(predicted=predicted, observed=observed)
    if np.isnan(metric) or np.isinf(metric):
        metric = 1e10
    log_prior = function.parameters.PDF(log=True) if prior else 0.0
    return -metric, log_prior, predicted, observed


def normalise_metrics(metrics):
    """
    Turn whatever a user supplied as `metrics` into a name-to-callable mapping.

    Parameters
    ----------
    metrics : dict, iterable, callable or None
        A mapping of output names to callables, an iterable of callables named
        after themselves, a single callable, or `None` for no extra metrics.

    Returns
    -------
    dict
        The metrics keyed by the name their values are recorded under.

    Raises
    ------
    TypeError
        If an entry is not callable, since it could not be evaluated later, when
        the error would be much harder to trace back to the call that caused it.
    """
    if metrics is None:
        return {}
    if callable(metrics):
        metrics = [metrics]
    if not isinstance(metrics, dict):
        metrics = {
            getattr(metric, "__name__", f"metric_{index}"): metric
            for index, metric in enumerate(metrics)
        }
    for name, metric in metrics.items():
        if not callable(metric):
            raise TypeError(
                f"The metric {name!r} is not callable. Metrics are called with "
                "keyword arguments, so each one has to be a function."
            )
    return dict(metrics)


def fit_extras(prior=False, metrics=None):
    """
    The names `evaluate_fit` adds to a fit, in the order it adds them.

    The optimisers that pack their results positionally need the names and the
    values in the same order, so both come from here rather than being written
    out twice.

    Parameters
    ----------
    prior : bool
        Whether the fit included the prior.
    metrics : dict, iterable, callable or None
        The user-supplied metrics.

    Returns
    -------
    list of str
        The names, in order.
    """
    names = ["log_likelihood", "log_prior"] if prior else []
    return names + list(normalise_metrics(metrics))


def evaluate_fit(pars, function, data, loss, prior=False, metrics=None):
    """
    Evaluate the quantities recorded alongside a fit at `pars`.

    Parameters
    ----------
    pars : array-like
        The parameter values to evaluate, normally the optimised ones.
    function : cpm.generators.Wrapper
        The model, as passed to `objective`.
    data
        The observed data, as passed to `objective`.
    loss : callable
        The loss function, as passed to `objective`.
    prior : bool
        Whether the fit included the prior. When `True`, the summed log
        likelihood and the summed log prior density are recorded, because `fun`
        is then the negative summed log posterior density and cannot be split
        apart afterwards.
    metrics : dict, iterable, callable or None
        Goodness-of-fit metrics to evaluate. Each is called with keyword
        arguments only, so a metric takes the ones it needs and absorbs the rest
        in `**kwargs`, as `cpm.optimisation.compare.PenalisedLikelihoods` and the
        loss functions in `cpm.optimisation.minimise` already do. The arguments
        supplied are `likelihood` and `log_likelihood` (both the summed log
        likelihood), `log_prior`, `predicted`, `observed`, `n` (the number of
        observations for this participant), `k` (the number of free parameters)
        and `parameters` (the model's `Parameters` object at the optimum).

    Returns
    -------
    dict
        The values to record, keyed as `fit_extras` lists them.

    Raises
    ------
    ValueError
        If a metric is named after a quantity already being recorded.

    Notes
    -----
    `n` is the length of the first axis of the observed data, which is the number
    of trials. A metric needing a different count can take `observed` and derive
    its own.
    """
    log_likelihood, log_prior, predicted, observed = _evaluate(
        pars, function, data, loss, prior
    )

    output = {}
    if prior:
        output["log_likelihood"] = log_likelihood
        output["log_prior"] = log_prior

    metrics = normalise_metrics(metrics)
    if metrics:
        arguments = {
            "likelihood": log_likelihood,
            "log_likelihood": log_likelihood,
            "log_prior": log_prior,
            "predicted": predicted,
            "observed": observed,
            "n": np.asarray(observed).shape[0],
            "k": len(np.asarray(pars).ravel()),
            "parameters": function.parameters,
        }
        for name, metric in metrics.items():
            if name in output:
                raise ValueError(
                    f"The metric {name!r} would overwrite the {name!r} recorded for "
                    "a fit with a prior. Metric names become column names in "
                    "`export()`, so give it a different one."
                )
            output[name] = metric(**arguments)
    return output


def prepare_data(data, identifier):
    """
    The function extracts variables from data and converts the data to a right structure before fitting.

    Parameters
    ----------
    data : pd.DataFrame, pd.DataFrameGroupBy, list
        The `data` parameter is the data that is used to fit the model.
    identifier : str
        The `identifier` parameter is the column name that is used to group the data.
    Returns
    -------
        The data, participants, groups and __pandas__ are being returned.
    """

    if isinstance(data, pd.api.typing.DataFrameGroupBy):
        groups = list(data.groups.keys())
        participants = data.get_group(groups[0])
        __pandas__ = True
    if isinstance(data, list):
        participants = data[0]
        groups = None
        __pandas__ = False
    if isinstance(data, pd.DataFrame) and identifier is not None:
        data = data.groupby(identifier)
        groups = list(data.groups.keys())
        participants = data.get_group(groups[0])
        __pandas__ = True

    return data, participants, groups, __pandas__
