"""
Fast log densities for the prior distributions cpm builds.

`Parameters.PDF` is evaluated on every call of the objective function, and a
scipy `logpdf` call on a frozen distribution costs about 30 microseconds, most
of it argument handling for arrays. For a single point, and for the
distribution families that `Value` builds from a string (and the same families
when they are passed in as frozen scipy distributions), this module evaluates
the same expression as scipy, in the same order of floating-point operations,
with the constants that depend only on the distribution's arguments computed
once. The results are identical to scipy's, not only close.

The scipy object stays the prior: this module only caches an evaluator on it.
The evaluator is rebuilt whenever the distribution's arguments change, so
priors that are updated in place still give the right density. Anything the
fast path does not cover (other distributions, array-valued or invalid parameters, points
on or outside the boundary of the support, non-finite values) is passed on to
scipy unchanged.
"""

import math

import numpy as np
from scipy import special

__all__ = ["fast_logpdf"]

## attribute under which the evaluator is cached on the frozen distribution
_CACHE = "_cpm_fast_logpdf"

try:  # scipy's normalising constants, as its distributions compute them
    from scipy.stats._continuous_distns import _log_gauss_mass, _norm_pdf_logC
except ImportError:  # pragma: no cover - a scipy without them uses scipy throughout
    _log_gauss_mass = _norm_pdf_logC = None


def _constants(name, shapes):
    """The terms of scipy's `_logpdf` that depend only on the shape parameters."""
    if name in ("norm", "truncnorm"):
        if _norm_pdf_logC is None:
            raise ValueError("scipy internals not available")
        if name == "norm":
            return (_norm_pdf_logC,)
        return (_norm_pdf_logC, np.real(_log_gauss_mass(*shapes))[()])
    if name == "gamma":
        return (shapes[0] - 1.0, special.gammaln(shapes[0]))
    if name == "beta":
        return (shapes[0] - 1.0, shapes[1] - 1.0, special.betaln(shapes[0], shapes[1]))
    if name == "truncexpon":
        return (np.log(-special.expm1(-shapes[0])),)
    return ()  # uniform


## scipy's `_logpdf` of each family at the standardised point y, with the
## constants above; the order of operations is scipy's
_LOGPDF = {
    "norm": lambda y, c: -(y * y) / 2.0 - c[0],
    "truncnorm": lambda y, c: (-(y * y) / 2.0 - c[0]) - c[1],
    "uniform": lambda y, c: 0.0,
    "gamma": lambda y, c: special.xlogy(c[0], y) - y - c[1],
    "beta": lambda y, c: (special.xlog1py(c[1], -y) + special.xlogy(c[0], y)) - c[2],
    "truncexpon": lambda y, c: -y - c[0],
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
    """The log density of one frozen distribution with fixed arguments."""

    __slots__ = ("args", "kwds", "loc", "scale", "log_scale", "constants", "logpdf", "low", "high")

    def __init__(self, prior, name):
        self.args = tuple(prior.args)
        self.kwds = dict(prior.kwds)
        shapes, loc, scale = prior.dist._parse_args(*prior.args, **prior.kwds)
        self.loc, self.scale = np.float64(loc), np.float64(scale)
        shapes = tuple(np.float64(s) for s in shapes)
        self.log_scale = np.log(self.scale)
        self.constants = _constants(name, shapes)
        self.logpdf = _LOGPDF[name]
        self.low, self.high = _support(name, shapes)

    def valid(self, prior):
        try:
            return bool(self.kwds == prior.kwds and self.args == tuple(prior.args))
        except (TypeError, ValueError):  # array-valued arguments
            return False

    def __reduce__(self):
        ## the cache is not pickled (`logpdf` is a lambda); an unpickled prior rebuilds it
        return type(None), ()

    def __call__(self, x):
        y = (x - self.loc) / self.scale
        if self.low < y < self.high:
            return self.logpdf(y, self.constants) - self.log_scale
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
    if name not in _LOGPDF or not hasattr(prior, "kwds") or not hasattr(prior, "args"):
        try:
            setattr(prior, _CACHE, False)
        except AttributeError:
            pass
        return False
    try:
        shapes, loc, scale = dist._parse_args(*prior.args, **prior.kwds)
        if not all(np.ndim(v) == 0 for v in (*shapes, loc, scale)) or not scale > 0:
            return False
        if not np.all(dist._argcheck(*shapes)):  # scipy returns NaN for these
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
        The log density, equal to `prior.logpdf(x)`, or None if the caller has to
        ask `prior.logpdf` itself.
    """
    if not isinstance(x, (float, int, np.floating, np.integer)) or isinstance(x, bool):
        return None
    evaluator = _evaluator(prior)
    if evaluator is False:
        return None
    value = evaluator(np.float64(x))
    return None if value is None else np.float64(value)
