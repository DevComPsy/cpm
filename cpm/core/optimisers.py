import numpy as np
import pandas as pd
import numdifftools as nd
import copy

__all__ = [
    "objective",
    "decompose_objective",
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
    function.reset(parameters=pars)
    function.run()
    predicted = copy.deepcopy(function.dependent)
    observed = copy.deepcopy(data)
    metric = loss(predicted=predicted, observed=observed)
    del predicted, observed
    if np.isnan(metric) or np.isinf(metric):
        metric = 1e10
    log_prior = function.parameters.PDF(log=True) if prior else 0.0
    return -metric, log_prior


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
