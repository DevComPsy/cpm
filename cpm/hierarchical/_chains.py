"""The random starting priors of the chains of the hierarchical methods, and where their participant-wise fits start."""

import numpy as np


def positive_definite(hessians):
    """
    Whether each participant's Hessian is positive definite (finite, with a Cholesky decomposition).

    Parameters
    ----------
    hessians : array-like
        The Hessians of the negative log posterior density, shape (participants, parameters, parameters).

    Returns
    -------
    numpy.ndarray
        One boolean per participant.
    """
    out = np.zeros(len(hessians), dtype=bool)
    for i, hessian in enumerate(hessians):
        if np.all(np.isfinite(hessian)):
            try:
                np.linalg.cholesky(hessian)
                out[i] = True
            except np.linalg.LinAlgError:
                pass
    return out


def check_start(start):
    if start not in ("random", "prior_mean"):
        raise ValueError(f'start must be "random" or "prior_mean", not {start!r}.')
    return start


def start_from_prior_locations(optimiser):
    """
    Replace the first start of `optimiser` with the location of the current prior of each free parameter, clipped into its bounds.

    The other starts are the random guesses that `optimiser.reset()` drew, so the
    same random numbers are drawn as with random starts only.

    Parameters
    ----------
    optimiser : object
        An optimiser of `cpm.optimisation`, after `reset()`.
    """
    parameters = optimiser.model.parameters
    locations = [getattr(parameters, name).prior.kwds["loc"] for name in parameters.free()]
    lower, upper = parameters.bounds()
    optimiser.initial_guess[0] = np.clip(locations, lower, upper)


def starting_priors(names, bounds, priors):
    """
    Draw a random starting prior for each parameter, for every chain after the first.

    Within finite bounds, the mean is a Beta(2, 2) draw stretched over the bounds,
    and the standard deviation a Beta(2, 2) draw stretched over half of their range.
    A parameter with an infinite bound instead starts from a draw of its prior when
    the estimation started, with the standard deviation of that prior. All draws
    come from NumPy's global random generator, so `numpy.random.seed` reproduces them.

    Parameters
    ----------
    names : list
        The names of the free parameters.
    bounds : tuple
        The lower and upper bounds of the free parameters, in the order of `names`.
    priors : dict
        The prior of each free parameter when the estimation started.

    Returns
    -------
    dict
        The mean and standard deviation of each starting prior, for `Parameters.update_prior`.
    """
    updates = {}
    for i, name in enumerate(names):
        lower, upper = bounds[0][i], bounds[1][i]
        if np.isfinite(lower) and np.isfinite(upper):
            updates[name] = {
                "mean": lower + np.random.beta(a=2, b=2) * (upper - lower),
                "sd": np.random.beta(a=2, b=2) * (upper - lower) / 2,
            }
        else:
            updates[name] = {
                "mean": float(priors[name].rvs()),
                "sd": float(priors[name].std()),
            }
    return updates
