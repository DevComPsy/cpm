"""
`FminBound(hessian="finite_differences")`: the Hessian at the optimum by finite
differences that stay within the bounds, computed once per participant.
"""

import numdifftools as nd
import numpy as np
import pytest

import cpm.optimisation.fmin as fmin_module
from cpm.core.optimisers import numerical_hessian, objective
from cpm.optimisation import FminBound, minimise
from test.optimisation.test_fminbound_gradient import bandit, rlrw

LOSS = minimise.LogLikelihood.bernoulli
A = np.array([[4.0, 1.5, 0.2], [1.5, 0.7, -0.1], [0.2, -0.1, 2.0]])


def counted(f):
    calls = []

    def g(x):
        calls.append(np.array(x, copy=True))
        return f(x)

    return g, calls


def quadratic(x):
    return float(0.5 * x @ A @ x + np.sin(1.0))


@pytest.mark.parametrize("d", [2, 3])
def test_interior_hessian_of_a_quadratic(d):
    from cpm.core.optimisers import finite_difference_hessian

    f, calls = counted(lambda x: quadratic(np.pad(x, (0, 3 - d))))
    x = np.array([0.3, 2.0, -1.0])[:d]
    H = finite_difference_hessian(f, x, np.full(d, -10.0), np.full(d, 10.0))
    np.testing.assert_allclose(H, A[:d, :d], rtol=1e-4, atol=1e-6)
    assert len(calls) == d * d + d + 1


def test_hessian_on_an_upper_bound_stays_within_the_bounds():
    from cpm.core.optimisers import finite_difference_hessian

    f, calls = counted(quadratic)
    lower, upper = np.array([-10.0, 0.0, -10.0]), np.array([10.0, 2.0, 10.0])
    x = np.array([0.3, 2.0, -1.0])
    H = finite_difference_hessian(f, x, lower, upper)
    assert all(np.all((c >= lower) & (c <= upper)) for c in calls)
    assert len(calls) == 3 * 4 // 2 + 3 + 1
    np.testing.assert_allclose(H, A, rtol=1e-4, atol=1e-6)


def learners(participants=6, trials=120, alpha=0.3, temperature=4.0, seed=2):
    """A bandit with rewarding arms and the choices of RLRW, so that the estimates are inside the bounds."""
    rng = np.random.default_rng(seed)
    data = bandit(participants, trials, seed)
    reward_probability = np.array([0.2, 0.4, 0.6, 0.8])
    for side in ("left", "right"):
        data[f"reward_{side}"] = (rng.random(len(data)) < reward_probability[data[f"arm_{side}"] - 1]).astype(float)
    choices = []
    for _, g in data.groupby("ppt"):
        values = np.full(4, 0.25)
        for left, right, reward_left, reward_right in g[["arm_left", "arm_right", "reward_left", "reward_right"]].to_numpy():
            left, right = int(left) - 1, int(right) - 1
            p_right = 1 / (1 + np.exp(-temperature * (values[right] - values[left])))
            choice = int(rng.random() < p_right)
            arm, reward = (right, reward_right) if choice else (left, reward_left)
            values[arm] += alpha * (reward - values[arm])
            choices.append(choice)
    data["response"] = data["observed"] = choices
    return data


def test_hessian_of_rlrw_agrees_with_numdifftools_central():
    from cpm.core.optimisers import finite_difference_hessian

    data = learners()
    model = rlrw(data)
    fit = FminBound(model=model, data=data.groupby("ppt"), minimisation=LOSS, prior=True, number_of_starts=1,
                    initial_guess=[0.5, 5.0], approx_grad=True)
    fit.optimise()
    lower, upper = (np.asarray(b, dtype=float) for b in model.parameters.bounds())
    checked = 0
    for (_, participant), result in zip(data.groupby("ppt"), fit.fit):
        x = result["x"]
        ## numdifftools' own steps must stay within the bounds too
        if np.any(x <= lower + 0.05 * (upper - lower)) or np.any(x >= upper - 0.05 * (upper - lower)):
            continue
        model.reset(data=participant)
        observed = participant.observed.to_numpy()
        f = lambda z: objective(z, model, observed, LOSS, True)  # noqa: E731
        expected = nd.Hessian(f, method="central")(x)
        H = finite_difference_hessian(f, x, lower, upper)
        assert np.abs(H - expected).max() <= 1e-4 * np.abs(expected).max()
        checked += 1
    assert checked > 0


def fit(hessian, temperature_upper=10.0, starts=2, **kwargs):
    data = bandit()
    np.random.seed(3)
    optimiser = FminBound(
        model=rlrw(data, temperature_upper),
        data=data.groupby("ppt"),
        minimisation=LOSS,
        prior=True,
        number_of_starts=starts,
        approx_grad=True,
        hessian=hessian,
        **kwargs,
    )
    optimiser.optimise()
    return optimiser, data


def test_default_is_the_numdifftools_hessian_next_to_the_optimum():
    optimiser, data = fit("numdifftools")
    model = rlrw(data)
    for (_, participant), result in zip(data.groupby("ppt"), optimiser.fit):
        model.reset(data=participant)
        observed = participant.observed.to_numpy()
        expected = numerical_hessian(lambda z: objective(z, model, observed, LOSS, True), result["x"] + 1e-3)
        np.testing.assert_array_equal(result["hessian"], expected)


def test_only_the_hessian_changes():
    old, _ = fit("numdifftools")
    new, _ = fit("finite_differences")
    assert len(old.fit) == len(new.fit)
    for a, b in zip(old.fit, new.fit):
        assert set(a) == set(b)
        for key in a:
            if key != "hessian":
                np.testing.assert_array_equal(a[key], b[key])
    assert old.parameters == new.parameters


def test_hessian_and_evaluate_fit_once_per_participant(monkeypatch):
    hessians, evaluations = [], []
    original_hessian, original_evaluate = fmin_module.finite_difference_hessian, fmin_module.evaluate_fit

    def hessian(*args, **kwargs):
        hessians.append(1)
        return original_hessian(*args, **kwargs)

    def evaluate(*args, **kwargs):
        evaluations.append(1)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(fmin_module, "finite_difference_hessian", hessian)
    monkeypatch.setattr(fmin_module, "evaluate_fit", evaluate)
    optimiser, data = fit("finite_differences", starts=3)
    participants = data.ppt.nunique()
    assert len(hessians) == participants
    assert len(evaluations) == participants
    assert all(np.all(np.isfinite(r["hessian"])) for r in optimiser.fit)


def test_only_the_hessian_changes_in_parallel():
    old, _ = fit("numdifftools", parallel=True, cl=2)
    new, _ = fit("finite_differences", parallel=True, cl=2)
    for a, b in zip(old.fit, new.fit):
        np.testing.assert_array_equal(a["x"], b["x"])
        assert a["fun"] == b["fun"]
        assert np.all(np.isfinite(b["hessian"]))


def test_no_all_zero_hessian_on_an_upper_bound():
    old, _ = fit("numdifftools", temperature_upper=0.5)
    new, _ = fit("finite_differences", temperature_upper=0.5)
    on_bound = [i for i, r in enumerate(new.fit) if r["x"][1] == 0.5]
    assert on_bound
    assert any(np.all(old.fit[i]["hessian"] == 0) for i in on_bound)  # what this fixes
    for i in on_bound:
        assert not np.all(new.fit[i]["hessian"] == 0)
        np.linalg.cholesky(new.fit[i]["hessian"])


def test_unknown_hessian_is_an_error():
    with pytest.raises(ValueError, match="hessian"):
        fit("central")
