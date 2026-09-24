"""
The session versions of the built-in applications give the same results as
the per-trial applications, on both backends: the same dependent variable,
export (columns, types and values), objective function and simulations.
"""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from cpm.applications.decision_making import (
    PTSM, PTSM1992, PTSM2025, PTSM1992Session, PTSM2025Session, PTSMSession,
)
from cpm.applications.reinforcement_learning import (
    RLRW, HybridMBMF, HybridMBMFSession, RLRWSession,
)
from cpm.core import _jit
from cpm.core.optimisers import objective
from cpm.datasets import load_bandit_data, load_risky_choices
from cpm.generators import Simulator
from cpm.optimisation import FminBound, minimise

BACKENDS = ["python", pytest.param("numba", marks=pytest.mark.skipif(
    not _jit.JIT_ENABLED, reason="numba is not installed or disabled"))]
TOLERANCE = dict(rtol=1e-12, atol=1e-12)


@pytest.fixture(autouse=True)
def quiet():
    warnings.simplefilter("ignore")


def bandit(ppt=1):
    data = load_bandit_data()
    data = data[data.ppt == ppt].reset_index(drop=True)
    data["observed"] = data["response"]
    return data


def two_step(n=120, seed=0):
    rng = np.random.default_rng(seed)
    action = rng.integers(0, 2, n)
    return pd.DataFrame({
        "s1": rng.integers(0, 2, n),
        "stimuli_first": rng.integers(0, 2, n),
        "action": action,
        "s2": 1 - action,
        "reward": rng.integers(0, 10, n) / 9,
        "reward_0": rng.integers(0, 10, n) / 9,
        "reward_1": rng.integers(0, 10, n) / 9,
        "observed": action,
    })


def risky(ppt=1):
    data = load_risky_choices()
    data = data[data.ppt == ppt].reset_index(drop=True)
    data["observed"] = data["choice"].astype(int)
    return data


PT = {
    "alpha": [0.8, 1e-2, 5.0], "lambda_loss": [1.6, 1e-2, 5.0], "gamma": [0.7, 1e-2, 5.0],
    "temperature": [3.0, 1e-2, 15.0], "beta": [0.9, 0.0, 5.0], "delta": [0.6, 1e-2, 5.0],
    "eta": [0.1, -0.49, 0.49], "phi_gain": [0.2, -10.0, 10.0], "phi_loss": [-0.3, -10.0, 10.0],
}
HYBRID = [[2, 0, 5], [0.4, 0, 1], [0.6, 0, 1], [0.5, 0, 1], [0.3, -5, 5], [-0.2, -5, 5]]


def cubic(x=None, alpha=None, lambda_loss=None):
    return np.where(x >= 0, x**3 * alpha, lambda_loss * x**3)


## (name, per-trial class, session class, data, keyword arguments, seed or None)
CASES = [
    ("RLRW", RLRW, RLRWSession, bandit, dict(dimensions=4, parameters_settings=[[0.3, 0, 1], [4, 0, 10]])),
    ("RLRW-generate", RLRW, RLRWSession, bandit,
     dict(dimensions=4, parameters_settings=[[0.3, 0, 1], [4, 0, 10]], generate=True)),
    ("HybridMBMF", HybridMBMF, HybridMBMFSession, two_step, dict(parameters_settings=HYBRID)),
    ("HybridMBMF-q_init", HybridMBMF, HybridMBMFSession, two_step,
     dict(parameters_settings=HYBRID, q_init=4.5)),
    ("HybridMBMF-generate", HybridMBMF, HybridMBMFSession, two_step,
     dict(parameters_settings=HYBRID, generate=True)),
    ("PTSM", PTSM, PTSMSession, risky, dict(parameters_settings=PT)),
    ("PTSM-power", PTSM, PTSMSession, risky, dict(parameters_settings=PT, weighting="power")),
    ("PTSM-prelec", PTSM, PTSMSession, risky, dict(parameters_settings=PT, weighting="prelec")),
    ("PTSM-generate", PTSM, PTSMSession, risky, dict(parameters_settings=PT, generate=True)),
    ("PTSM1992", PTSM1992, PTSM1992Session, risky, dict(parameters_settings=PT)),
    ("PTSM1992-gw", PTSM1992, PTSM1992Session, risky, dict(parameters_settings=PT, weighting="gw")),
    ("PTSM1992-curve", PTSM1992, PTSM1992Session, risky, dict(parameters_settings=PT, utility_curve=cubic)),
    ("PTSM2025", PTSM2025, PTSM2025Session, risky, dict(parameters_settings=PT)),
    ("PTSM2025-standard", PTSM2025, PTSM2025Session, risky, dict(parameters_settings=PT, variant="standard")),
]
IDS = [case[0] for case in CASES]


def build(case, backend):
    _, trial_class, session_class, data, kwargs = case
    frame = data()
    return (trial_class(data=frame, **kwargs), session_class(data=frame, backend=backend, **kwargs),
            frame["observed"].to_numpy())


def free_values(model, scale):
    """Parameter values inside the bounds, away from the initial ones."""
    lower, upper = model.parameters.bounds()
    return lower + scale * (upper - lower)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_export_equals_the_per_trial_model(case, backend):
    trial, session, _ = build(case, backend)
    for scale in (None, 0.2, 0.7):
        x = None if scale is None else free_values(trial, scale)
        for model in (trial, session):
            np.random.seed(123)
            model.reset(parameters=x)
            model.run()
        expected, got = trial.export(), session.export()
        assert list(got.columns) == list(expected.columns)
        assert dict(got.dtypes) == dict(expected.dtypes)
        np.testing.assert_allclose(got.to_numpy(float), expected.to_numpy(float), **TOLERANCE)
        np.testing.assert_allclose(session.dependent, trial.dependent, **TOLERANCE)
        ## the states a run ends with
        for key in session.parameters.keys():
            value = session.parameters[key]
            numeric = isinstance(getattr(value, "value", None), (int, float, np.ndarray, np.number))
            if numeric and value.prior is None:
                np.testing.assert_allclose(np.asarray(value.value, float),
                                           np.asarray(trial.parameters[key].value, float), **TOLERANCE)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", [c for c in CASES if "generate" not in c[0]],
                         ids=[i for i in IDS if "generate" not in i])
def test_objective_equals_the_per_trial_model(case, backend):
    """
    The objective agrees to 1e-12 where the predicted probabilities are not
    extreme. Where a probability p is within about 1e-6 of 0 or 1, the loss takes
    log(1 - p), which turns the rounding error of p (of order 1e-16 in either
    implementation) into a relative error of order 1e-16 / (1 - p); there the
    two agree to within that bound.
    """
    trial, session, observed = build(case, backend)
    for scale in (0.1, 0.35, 0.5, 0.8):
        x = free_values(trial, scale)
        np.random.seed(1)
        expected = objective(x, trial, observed, minimise.LogLikelihood.bernoulli, True)
        np.random.seed(1)
        got = objective(x, session, observed, minimise.LogLikelihood.bernoulli, True)
        p = np.clip(trial.dependent, 1e-10, 1 - 1e-10)
        closest = np.min(np.minimum(p, 1 - p))
        rel = 1e-12 if closest > 1e-6 else 1e-12 + 1e-15 / closest
        assert got == pytest.approx(expected, rel=rel, abs=1e-12)


@pytest.mark.skipif(not _jit.JIT_ENABLED, reason="numba is not installed or disabled")
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_backends_agree(case):
    _, _, session_class, data, kwargs = case
    frame = data()
    exports = []
    for backend in ("python", "numba"):
        model = session_class(data=frame, backend=backend, **kwargs)
        np.random.seed(7)
        model.run()
        exports.append(model.export())
    pd.testing.assert_frame_equal(exports[0], exports[1], check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_simulations_are_reproducible_and_match_the_per_trial_model(backend):
    data = pd.concat([two_step(60, seed=s).assign(ppt=s) for s in range(3)], ignore_index=True)
    draws = pd.DataFrame({
        "inv_temperature": [1.0, 2.0, 3.0], "learning_rate": [0.2, 0.5, 0.8],
        "eligibility_trace": [0.5, 0.5, 0.5], "mb_weight": [0.1, 0.5, 0.9],
        "choice_stickiness": [0.0, 0.3, -0.3], "response_stickiness": [0.2, 0.0, 0.1],
    })
    out = []
    for cls, kwargs in ((HybridMBMF, {}), (HybridMBMFSession, {"backend": backend})):
        wrapper = cls(data=data[data.ppt == 0], parameters_settings=HYBRID, generate=True, **kwargs)
        simulator = Simulator(wrapper=wrapper, data=data.groupby("ppt"), parameters=draws)
        np.random.seed(2024)
        simulator.run()
        out.append(simulator.export())
    pd.testing.assert_frame_equal(out[1], out[0], check_exact=False, rtol=1e-12, atol=1e-12)


def test_unknown_backend_and_missing_numba(monkeypatch):
    with pytest.raises(ValueError, match="backend"):
        RLRWSession(data=bandit(), dimensions=4, backend="cuda")
    monkeypatch.setattr(_jit, "HAVE_NUMBA", False)
    monkeypatch.setattr(_jit, "JIT_ENABLED", False)
    with pytest.raises(ImportError, match="numba"):
        RLRWSession(data=bandit(), dimensions=4, backend="numba")
    assert RLRWSession(data=bandit(), dimensions=4, backend="auto").backend == "python"


def test_missing_columns_are_named():
    with pytest.raises(KeyError, match="ambiguity"):
        PTSM2025Session(data=risky().drop(columns="ambiguity"), parameters_settings=PT)
    with pytest.raises(KeyError, match="reward_0"):
        HybridMBMFSession(data=two_step().drop(columns="reward_0"), generate=True)


def test_session_models_survive_deepcopy_and_pickling():
    import pickle

    model = HybridMBMFSession(data=two_step(), parameters_settings=HYBRID)
    model.run()
    clone = pickle.loads(pickle.dumps(copy.deepcopy(model)))
    clone.reset()
    clone.run()
    np.testing.assert_array_equal(clone.dependent, model.dependent)


@pytest.mark.skipif(not _jit.JIT_ENABLED, reason="numba is not installed or disabled")
def test_parallel_fit_matches_the_serial_fit():
    """With multiprocess on Windows (spawn), each worker loads the cached kernels."""
    data = pd.concat([bandit(p).assign(ppt=p) for p in (1, 2, 3, 4)], ignore_index=True)
    fits = []
    for parallel in (False, True):
        fit = FminBound(
            model=RLRWSession(data=bandit(1), dimensions=4, parameters_settings=[[0.3, 0, 1], [4, 0, 10]],
                              backend="numba"),
            data=data,
            minimisation=minimise.LogLikelihood.bernoulli,
            prior=True,
            ppt_identifier="ppt",
            initial_guess=[[0.4, 3.0]],
            number_of_starts=1,
            approx_grad=True,
            parallel=parallel,
            cl=2 if parallel else None,
        )
        fit.optimise()
        fits.append(fit.export().sort_values("ppt").reset_index(drop=True))
    for column in ("x_0", "x_1", "fun"):
        np.testing.assert_allclose(fits[1][column], fits[0][column], rtol=1e-10)
