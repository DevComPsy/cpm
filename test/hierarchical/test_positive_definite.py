"""
The hierarchical methods record, per participant and iteration, whether the
Hessian of the negative log posterior is positive definite, and per iteration how
many are not. An estimate on a bound can have a Hessian that is not; it is kept.
"""

import warnings

import numpy as np
import pytest

from cpm.hierarchical import EmpiricalBayes, VariationalBayes
from cpm.optimisation import FminBound, minimise
from test.optimisation.test_fminbound_gradient import bandit, rlrw

METHODS = [EmpiricalBayes, VariationalBayes]


def method(cls, hessian):
    warnings.simplefilter("ignore")
    data = bandit(participants=6, trials=40)
    ## with an upper bound of 1 on the temperature, estimates end on it, where the
    ## "numdifftools" Hessian is all zeros (with 0.5, EmpiricalBayes fails on
    ## these all-zero Hessians, as before)
    optimiser = FminBound(model=rlrw(data, temperature_upper=1.0), data=data.groupby("ppt"),
                          minimisation=minimise.LogLikelihood.bernoulli, prior=True, number_of_starts=1,
                          approx_grad=True, hessian=hessian)
    tolerance = {"tolerance": 0.0} if cls is EmpiricalBayes else {"tolerance_lme": 0.0, "tolerance_param": 0.0}
    fit = cls(optimiser=optimiser, iteration=3, chain=1, quiet=True, **tolerance)
    hessians = []
    optimise = fit.optimiser.optimise

    def recorded(*args, **kwargs):
        result = optimise(*args, **kwargs)
        hessians.append([r["hessian"] for r in fit.optimiser.fit])
        return result

    fit.optimiser.optimise = recorded
    np.random.seed(2)
    fit.optimise()
    return fit, hessians


def is_positive_definite(h):
    if not np.all(np.isfinite(h)):
        return False
    try:
        np.linalg.cholesky(h)
        return True
    except np.linalg.LinAlgError:
        return False


@pytest.mark.parametrize("hessian", ["numdifftools", "finite_differences"])
@pytest.mark.parametrize("cls", METHODS)
def test_flags_and_counts(cls, hessian):
    fit, hessians = method(cls, hessian)
    expected = np.array([[is_positive_definite(h) for h in step] for step in hessians])
    flags = fit.fit.sort_values(["iteration", "ppt"]).positive_definite.to_numpy().reshape(expected.shape)
    np.testing.assert_array_equal(flags, expected)
    counts = fit.hyperparameters.groupby("iteration").not_positive_definite.first().to_numpy()
    np.testing.assert_array_equal(counts, (~expected).sum(axis=1))
    if hessian == "numdifftools":
        assert counts.sum() > 0  # the all-zero Hessians on the bound


@pytest.mark.parametrize("cls", METHODS)
def test_estimates_on_a_bound_are_kept(cls):
    fit, _ = method(cls, "numdifftools")
    flagged = fit.fit[~fit.fit.positive_definite]
    assert len(flagged) > 0
    ## each with a parameter on a bound: alpha in [0.01, 1], temperature in [0, 1]
    assert (flagged.alpha.isin([0.01, 1.0]) | flagged.temperature.isin([0.0, 1.0])).all()
