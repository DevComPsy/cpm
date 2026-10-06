"""
`FminBound` with `approx_grad=True` computes the finite-difference gradient
itself, together with the objective, instead of leaving it to SciPy. These tests
check that the fits are exactly the ones SciPy's own `approx_grad` gives.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import fmin_l_bfgs_b

from cpm.applications.reinforcement_learning import RLRW
from cpm.core.data import decompose
from cpm.core.optimisers import objective
from cpm.optimisation import FminBound, minimise

LOSS = minimise.LogLikelihood.bernoulli


def bandit(participants=4, trials=40, seed=1):
    rng = np.random.default_rng(seed)
    arms = np.array([rng.choice(4, size=2, replace=False) for _ in range(participants * trials)]) + 1
    return pd.DataFrame(
        {
            "ppt": np.repeat(np.arange(1, participants + 1), trials),
            "arm_left": arms[:, 0],
            "arm_right": arms[:, 1],
            "reward_left": rng.integers(0, 2, participants * trials).astype(float),
            "reward_right": rng.integers(0, 2, participants * trials).astype(float),
            "response": rng.integers(0, 2, participants * trials),
            "observed": rng.integers(0, 2, participants * trials),
        }
    )


def rlrw(data, temperature_upper=10.0):
    model = RLRW(
        data=data[data.ppt == 1],
        dimensions=4,
        parameters_settings=[[0.5, 1e-2, 1.0], [2.0, 0.0, temperature_upper]],
    )
    model.parameters.update_prior(alpha={"mean": 0.5, "sd": 0.5}, temperature={"mean": 5.0, "sd": 5.0})
    return model


def scipy_reference(model, data, guess, prior, **options):
    """What `fmin_l_bfgs_b` with SciPy's own `approx_grad` gives for every participant."""
    bounds = list(map(tuple, np.asarray(model.parameters.bounds()).T))
    out = []
    for participant in data.groupby("ppt"):
        participant_dc, observed, _ = decompose(participant=participant, pandas=True, identifier="ppt")
        model.reset(data=participant_dc)
        out.append(
            fmin_l_bfgs_b(objective, x0=guess, bounds=bounds, args=(model, observed, LOSS, prior), approx_grad=True, **options)
        )
    return out


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("temperature_upper", [10.0, 0.5])
@pytest.mark.parametrize("options", [{}, {"epsilon": 1e-6}, {"maxfun": 10}])
def test_fits_are_identical_to_scipy_approx_grad(prior, temperature_upper, options):
    data = bandit()
    guess = np.array([0.3, 0.4])
    fit = FminBound(
        model=rlrw(data, temperature_upper),
        data=data.groupby("ppt"),
        minimisation=LOSS,
        prior=prior,
        initial_guess=guess,
        number_of_starts=1,
        approx_grad=True,
        **options,
    )
    fit.optimise()
    reference = scipy_reference(rlrw(data, temperature_upper), data, guess, prior, **options)
    for got, (x, f, info) in zip(fit.fit, reference):
        np.testing.assert_array_equal(got["x"], x)
        assert got["fun"] == f
        np.testing.assert_array_equal(got["grad"], info["grad"])
        assert got["funcalls"] == info["funcalls"]
        assert got["nit"] == info["nit"]
        assert got["warnflag"] == info["warnflag"]
        assert got["task"] == info["task"]


def test_a_participant_on_the_upper_bound_is_covered():
    """With an upper bound of 0.5 on the temperature, the estimates end on it, so the step must flip sign."""
    data = bandit()
    reference = scipy_reference(rlrw(data, 0.5), data, np.array([0.3, 0.4]), True)
    assert any(x[1] == 0.5 for x, _, _ in reference)


@pytest.mark.parametrize(
    "x, lower, upper",
    [
        ([0.4, 3.0], [1e-2, 0.0], [1.0, 10.0]),  # inside
        ([1e-2, 10.0], [1e-2, 0.0], [1.0, 10.0]),  # on a lower and an upper bound
        ([0.4, -3.0], [-np.inf, -np.inf], [np.inf, np.inf]),  # unbounded
        ([0.5, 1.0], [0.5, 1.0 - 4e-9], [0.5 + 3e-9, 1.0]),  # bounds closer than the step
        ([1e9, -1e9], [-np.inf, -2e9], [2e9, np.inf]),  # a step too small to change x
    ],
)
def test_gradient_is_scipys(x, lower, upper):
    from scipy.optimize._numdiff import approx_derivative

    from cpm.optimisation.fmin import gradient_by_forward_differences

    def f(z):
        return float(np.sin(z[0]) * z[1] + z[1] ** 2 / 3)

    x, lower, upper = (np.asarray(v, dtype=float) for v in (x, lower, upper))
    expected = approx_derivative(f, x, method="2-point", abs_step=1e-8, f0=f(x), bounds=(lower, upper))
    np.testing.assert_array_equal(gradient_by_forward_differences(f, x, f(x), lower, upper, 1e-8), expected)


def test_gradient_steps_away_from_a_bound():
    from cpm.optimisation.fmin import gradient_by_forward_differences

    calls = []

    def f(x):
        calls.append(x.copy())
        return float(np.sum(x**2))

    lower, upper = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    x = np.array([1.0, 0.5])
    gradient = gradient_by_forward_differences(f, x, f(x), lower, upper, 1e-8)
    assert all(np.all((c >= lower) & (c <= upper)) for c in calls)
    assert calls[1][0] < 1.0  # stepped down from the upper bound
    np.testing.assert_allclose(gradient, 2 * x, rtol=1e-6)
