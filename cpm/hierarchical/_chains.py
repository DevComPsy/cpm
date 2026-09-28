"""The random starting priors of the chains of the hierarchical methods."""

import numpy as np


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
