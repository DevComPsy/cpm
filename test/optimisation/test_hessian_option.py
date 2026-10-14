"""
The `hessian` option of every optimiser: "numdifftools" (the default, the previous
Hessian) or "finite_differences" (at the optimum, within the bounds, once per
participant for the best start).
"""

import numpy as np
import pytest

import cpm.optimisation.fmin as fmin_module
import cpm.optimisation.free as free_module
from cpm.core.optimisers import finite_difference_hessian, objective
from cpm.optimisation import Fmin, FminBound, Minimize, minimise
from test.optimisation.test_fminbound_gradient import bandit, rlrw

try:
    import cpm.optimisation.bads as bads_module
    from cpm.optimisation import Bads

    HAS_BADS = True
except ImportError:  # pragma: no cover - pybads is an optional dependency
    HAS_BADS = False

LOSS = minimise.LogLikelihood.bernoulli
OPTIMISERS = [Fmin, FminBound, Minimize]
MODULES = {Fmin: fmin_module, FminBound: fmin_module, Minimize: free_module}
if HAS_BADS:
    OPTIMISERS.append(Bads)
    MODULES[Bads] = bads_module
IDS = [cls.__name__ for cls in OPTIMISERS]
PARTICIPANTS = 3
## Bads records its own run time
VARYING = {"total_time"}


def build(cls, hessian=None):
    data = bandit(participants=PARTICIPANTS, trials=30)
    kwargs = {} if hessian is None else {"hessian": hessian}
    if cls is FminBound:
        kwargs["approx_grad"] = True
    if HAS_BADS and cls is Bads:
        kwargs["options"] = {"display": "off"}
    np.random.seed(4)
    optimiser = cls(model=rlrw(data), data=data.groupby("ppt"), minimisation=LOSS, prior=True, number_of_starts=2, **kwargs)
    optimiser.optimise()
    return optimiser, data


def records(optimiser):
    """The per-participant records that hold the Hessian: `fit`, except for `Minimize`, which keeps it in `details`."""
    return optimiser.details if isinstance(optimiser, Minimize) else optimiser.fit


def assert_same(a, b, skip=()):
    assert set(a) == set(b)
    for key in set(a) - set(skip) - VARYING:
        np.testing.assert_array_equal(np.asarray(a[key], dtype=object), np.asarray(b[key], dtype=object), err_msg=key)


@pytest.mark.parametrize("cls", OPTIMISERS, ids=IDS)
def test_default_is_numdifftools(cls):
    default, _ = build(cls)
    explicit, _ = build(cls, "numdifftools")
    assert default.hessian == "numdifftools"
    for a, b in zip(default.fit, explicit.fit):
        assert_same(a, b)


@pytest.mark.parametrize("cls", OPTIMISERS, ids=IDS)
def test_only_the_hessian_changes(cls):
    old, _ = build(cls, "numdifftools")
    new, _ = build(cls, "finite_differences")
    for a, b in zip(old.fit, new.fit):
        assert_same(a, b, skip={"hessian"})
    for a, b in zip(records(old), records(new)):
        assert not np.array_equal(a["hessian"], b["hessian"])
    assert old.parameters == new.parameters


@pytest.mark.parametrize("cls", OPTIMISERS, ids=IDS)
def test_hessian_is_the_finite_difference_one_at_the_optimum(cls):
    fit, data = build(cls, "finite_differences")
    model = rlrw(data)
    lower, upper = (np.asarray(b, dtype=float) for b in model.parameters.bounds())
    for (_, participant), record in zip(data.groupby("ppt"), records(fit)):
        model.reset(data=participant)
        observed = participant.observed.to_numpy()
        x = record["xopt"] if "xopt" in record else record["x"]  # Fmin records its estimates as xopt
        expected = finite_difference_hessian(lambda z: objective(z, model, observed, LOSS, True), x, lower, upper)
        np.testing.assert_array_equal(record["hessian"], expected)


@pytest.mark.parametrize("cls", OPTIMISERS, ids=IDS)
def test_hessian_and_extras_once_per_participant(cls, monkeypatch):
    module = MODULES[cls]
    calls = {"hessian": 0, "evaluate_fit": 0}
    original_hessian, original_evaluate = module.finite_difference_hessian, module.evaluate_fit

    def hessian(*args, **kwargs):
        calls["hessian"] += 1
        return original_hessian(*args, **kwargs)

    def evaluate(*args, **kwargs):
        calls["evaluate_fit"] += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(module, "finite_difference_hessian", hessian)
    monkeypatch.setattr(module, "evaluate_fit", evaluate)
    build(cls, "finite_differences")
    assert calls == {"hessian": PARTICIPANTS, "evaluate_fit": PARTICIPANTS}


@pytest.mark.parametrize("cls", OPTIMISERS, ids=IDS)
def test_unknown_hessian_is_an_error(cls):
    with pytest.raises(ValueError, match="hessian"):
        build(cls, "central")
