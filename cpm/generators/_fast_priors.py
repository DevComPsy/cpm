"""
Closed-form log densities for the prior distributions cpm builds.

`Parameters.PDF` is evaluated on every call of the objective function, and a
scipy `logpdf` call on a frozen distribution costs about 30 microseconds, most
of it argument checking. For the distribution families that `Value` builds
from a string (and the same families when they are passed in as frozen scipy
distributions), the log density at a point inside the support is a short
closed-form kernel plus a constant. This module evaluates that instead.

The scipy object stays the prior: this module only caches an evaluator on it.
The evaluator is rebuilt whenever the distribution's arguments change, so
priors that are updated in place still give the right density. Anything the
fast path does not cover (other distributions, array-valued parameters, points
on or outside the boundary of the support, non-finite values) is passed on to
scipy unchanged.
"""

import math

import numpy as np

__all__ = ["fast_logpdf"]

## attribute under which the evaluator is cached on the frozen distribution
_CACHE = "_cpm_fast_logpdf"

## kernels of the log density in the standardised variable y = (x - loc) / scale,
## up to a constant, keyed by scipy distribution name
_KERNELS = {
    "norm": lambda y, shapes: -0.5 * y * y,
    "truncnorm": lambda y, shapes: -0.5 * y * y,
    "uniform": lambda y, shapes: 0.0,
    "gamma": lambda y, shapes: (shapes[0] - 1.0) * math.log(y) - y,
    "beta": lambda y, shapes: (shapes[0] - 1.0) * math.log(y)
    + (shapes[1] - 1.0) * math.log1p(-y),
    "truncexpon": lambda y, shapes: -y,
}


def _support(name, shapes):
    """The open interior of the support in the standardised variable."""
    if name == "norm":
        return -math.inf, math.inf
    if name == "truncnorm":
        return shapes[0], shapes[1]
    if name == "gamma":
        return 0.0, math.inf
    if name == "truncexpon":
        return 0.0, shapes[0]
    return 0.0, 1.0  # uniform, beta


class _Evaluator:
    """The closed-form log density of one frozen distribution with fixed arguments."""

    __slots__ = ("args", "kwds", "loc", "scale", "shapes", "kernel", "low", "high", "const")

    def __init__(self, prior, name):
        self.args = tuple(prior.args)
        self.kwds = dict(prior.kwds)
        shapes, loc, scale = prior.dist._parse_args(*prior.args, **prior.kwds)
        self.loc, self.scale = float(loc), float(scale)
        self.shapes = tuple(float(s) for s in shapes)
        self.kernel = _KERNELS[name]
        self.low, self.high = _support(name, self.shapes)
        ## the normalising constant, taken from scipy at the median, which lies
        ## inside the support for all of the families above
        median = float(prior.median())
        reference = float(prior.logpdf(median))
        y = (median - self.loc) / self.scale
        if not (self.low < y < self.high) or not math.isfinite(reference):
            raise ValueError("no interior reference point")
        self.const = reference - self.kernel(y, self.shapes)

    def valid(self, prior):
        try:
            return bool(self.kwds == prior.kwds and self.args == tuple(prior.args))
        except (TypeError, ValueError):  # array-valued arguments
            return False

    def __call__(self, x):
        y = (x - self.loc) / self.scale
        if self.low < y < self.high:
            return self.kernel(y, self.shapes) + self.const
        return None


def _evaluator(prior):
    """
    The cached evaluator of `prior`, (re)built if its arguments changed.

    Returns False if there is no fast path for `prior`. Only an unsupported
    distribution family is remembered as such, since everything else depends on
    arguments that may still change.
    """
    cached = getattr(prior, _CACHE, None)
    if cached is False:
        return False
    if cached is not None and cached.valid(prior):
        return cached
    dist = getattr(prior, "dist", None)
    name = getattr(dist, "name", None)
    if name not in _KERNELS or not hasattr(prior, "kwds") or not hasattr(prior, "args"):
        try:
            setattr(prior, _CACHE, False)
        except AttributeError:
            pass
        return False
    try:
        shapes, loc, scale = dist._parse_args(*prior.args, **prior.kwds)
        if not all(np.ndim(v) == 0 for v in (*shapes, loc, scale)) or not scale > 0:
            return False
        evaluator = _Evaluator(prior, name)
    except Exception:
        return False
    try:
        setattr(prior, _CACHE, evaluator)
    except AttributeError:
        return False
    return evaluator


def fast_logpdf(prior, x):
    """
    The log density of `prior` at `x`, or None where the fast path does not apply.

    Parameters
    ----------
    prior : object
        The prior distribution of a `Value`, normally a frozen scipy distribution.
    x : float
        The point at which to evaluate the log density.

    Returns
    -------
    numpy.float64 or None
        The log density, equal to `prior.logpdf(x)` to within rounding error, or
        None if the caller has to ask `prior.logpdf` itself.
    """
    if not isinstance(x, (float, int, np.floating, np.integer)) or isinstance(x, bool):
        return None
    evaluator = _evaluator(prior)
    if evaluator is False:
        return None
    value = evaluator(float(x))
    return None if value is None else np.float64(value)
