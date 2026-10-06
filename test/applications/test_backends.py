"""
The built-in applications give the same results as their former per-trial
implementations, with and without numba.

The applications compute all trials of a participant in one call, compiled with
numba if it is installed. Until commit 3385574 they were per-trial `Wrapper`
models built from the classes in `cpm.models`. The outputs of those per-trial
implementations, for the cases below, are frozen in
`data/per_trial_reference.pkl.gz`: exports, dependent variables, final states,
objective values, the per-trial `simulation` records, a call of the per-trial
`model` function, and a seeded simulation with `Simulator`. Tables are stored as
dictionaries of numpy arrays, so that any version of pandas can read them.

numba and NumPy may compute exponentials differently in the last digit, depending
on the platform and CPU, so results are compared to a relative tolerance of 1e-12.
"""

import builtins
import copy
import gzip
import os
import pathlib
import pickle
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest

from cpm.applications.decision_making import PTSM, PTSM1992, PTSM2025
from cpm.applications.reinforcement_learning import RLRW, HybridMBMF
from cpm.core import _jit
from cpm.core.data import unpack_trials
from cpm.core.optimisers import objective
from cpm.datasets import load_bandit_data, load_risky_choices
from cpm.generators import Simulator, Wrapper
from cpm.optimisation import FminBound, minimise

with gzip.open(os.path.join(os.path.dirname(__file__), "data", "per_trial_reference.pkl.gz")) as f:
    REFERENCE = pickle.load(f)

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


## (name in the reference, class, data, keyword arguments)
CASES = [
    ("RLRW", RLRW, bandit, dict(dimensions=4, parameters_settings=[[0.3, 0, 1], [4, 0, 10]])),
    ("RLRW-generate", RLRW, bandit,
     dict(dimensions=4, parameters_settings=[[0.3, 0, 1], [4, 0, 10]], generate=True)),
    ("HybridMBMF", HybridMBMF, two_step, dict(parameters_settings=HYBRID)),
    ("HybridMBMF-q_init", HybridMBMF, two_step, dict(parameters_settings=HYBRID, q_init=4.5)),
    ("HybridMBMF-generate", HybridMBMF, two_step, dict(parameters_settings=HYBRID, generate=True)),
    ("PTSM", PTSM, risky, dict(parameters_settings=PT)),
    ("PTSM-power", PTSM, risky, dict(parameters_settings=PT, weighting="power")),
    ("PTSM-prelec", PTSM, risky, dict(parameters_settings=PT, weighting="prelec")),
    ("PTSM-generate", PTSM, risky, dict(parameters_settings=PT, generate=True)),
    ("PTSM1992", PTSM1992, risky, dict(parameters_settings=PT)),
    ("PTSM1992-gw", PTSM1992, risky, dict(parameters_settings=PT, weighting="gw")),
    ("PTSM1992-curve", PTSM1992, risky, dict(parameters_settings=PT, utility_curve=cubic)),
    ("PTSM2025", PTSM2025, risky, dict(parameters_settings=PT)),
    ("PTSM2025-standard", PTSM2025, risky, dict(parameters_settings=PT, variant="standard")),
]
IDS = [case[0] for case in CASES]
## outputs added since the reference was frozen, tested separately
ADDED = {RLRW: ["response"]}


def without_added(case, output):
    """`output` (an export or a per-trial record) without the outputs added since the reference."""
    added = ADDED.get(case[1], [])
    if isinstance(output, pd.DataFrame):
        return output.drop(columns=added)
    return {key: value for key, value in output.items() if key not in added}


def on_backend(model, backend):
    """Run `model` on `backend` (normally chosen by whether numba is installed)."""
    model._session_model.backend = backend
    return model


def build(case, backend):
    _, cls, data, kwargs = case
    frame = data()
    return on_backend(cls(data=frame, **kwargs), backend), frame["observed"].to_numpy()


def free_values(model, scale):
    """Parameter values inside the bounds, away from the initial ones."""
    lower, upper = model.parameters.bounds()
    return lower + scale * (upper - lower)


def assert_same_records(got, expected):
    assert list(got) == list(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(expected[key], float), **TOLERANCE)
        assert np.shape(got[key]) == np.shape(expected[key]), key


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_runs_equal_the_per_trial_implementation(case, backend):
    reference = REFERENCE["cases"][case[0]]
    model, _ = build(case, backend)
    for scale in (None, 0.2, 0.7):
        x = None if scale is None else free_values(model, scale)
        np.random.seed(123)
        model.reset(parameters=x)
        model.run()
        np.testing.assert_allclose(model.dependent, reference["dependent"][scale], **TOLERANCE)
        if scale in reference["exports"]:
            expected, got = pd.DataFrame(reference["exports"][scale]), without_added(case, model.export())
            assert list(got.columns) == list(expected.columns)
            assert dict(got.dtypes) == dict(expected.dtypes)
            np.testing.assert_allclose(got.to_numpy(float), expected.to_numpy(float), **TOLERANCE)
        if scale is None:
            for key, value in reference["states"].items():
                np.testing.assert_allclose(np.asarray(model.parameters[key].value, float), value, **TOLERANCE)
            ## the per-trial records of the run, as a per-trial Wrapper kept them
            assert len(model.simulation) == len(model.dependent)
            for got, expected in zip(model.simulation[:3], reference["simulation"]):
                assert_same_records(without_added(case, got), expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", [c for c in CASES if "generate" not in c[0]],
                         ids=[i for i in IDS if "generate" not in i])
def test_objective_equals_the_per_trial_implementation(case, backend):
    model, observed = build(case, backend)
    for scale, (expected, closest) in REFERENCE["cases"][case[0]]["objective"].items():
        np.random.seed(1)
        got = objective(free_values(model, scale), model, observed, minimise.LogLikelihood.bernoulli, True)
        ## near p = 0 or 1, rounding errors in p are amplified by 1 / (1 - p) in the log likelihood
        rel = 1e-12 if closest > 1e-6 else 1e-12 + 1e-15 / closest
        assert got == pytest.approx(expected, rel=rel, abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_per_trial_model_function_is_unchanged(case, backend):
    """`model(parameters, trial)` computes one trial from the states in `parameters`."""
    model, _ = build(case, backend)
    np.random.seed(5)
    first = unpack_trials(model.data, 0, model.__pandas__)
    got = model.model(parameters=model.parameters, trial=first)
    assert_same_records(without_added(case, got), REFERENCE["cases"][case[0]]["model"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_per_trial_model_function_reproduces_a_run(case, backend):
    """Wrapping `model` in a per-trial Wrapper, as users do to extend a model, gives the same run."""
    model, _ = build(case, backend)
    per_trial = Wrapper(model=model.model, data=model.data, parameters=model.parameters)
    for wrapper in (model, per_trial):
        np.random.seed(9)
        wrapper.run()
    pd.testing.assert_frame_equal(per_trial.export(), model.export(), check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_simulations_equal_the_per_trial_implementation(backend):
    data = pd.concat([two_step(60, seed=s).assign(ppt=s) for s in range(3)], ignore_index=True)
    draws = pd.DataFrame({
        "inv_temperature": [1.0, 2.0, 3.0], "learning_rate": [0.2, 0.5, 0.8],
        "eligibility_trace": [0.5, 0.5, 0.5], "mb_weight": [0.1, 0.5, 0.9],
        "choice_stickiness": [0.0, 0.3, -0.3], "response_stickiness": [0.2, 0.0, 0.1],
    })
    wrapper = on_backend(HybridMBMF(data=data[data.ppt == 0], parameters_settings=HYBRID, generate=True), backend)
    simulator = Simulator(wrapper=wrapper, data=data.groupby("ppt"), parameters=draws)
    np.random.seed(2024)
    simulator.run()
    pd.testing.assert_frame_equal(simulator.export(), pd.DataFrame(REFERENCE["simulator"]), check_exact=False,
                                  rtol=1e-12, atol=1e-12)


def simulate_rlrw(backend, seed=0):
    data = pd.concat([bandit(p).assign(ppt=p) for p in (1, 2, 3)], ignore_index=True)
    draws = pd.DataFrame({"alpha": [0.3, 0.5, 0.7], "temperature": [2.0, 5.0, 8.0]})
    wrapper = on_backend(RLRW(data=bandit(1), dimensions=4, parameters_settings=[[0.5, 0, 1], [5, 0, 10]],
                              generate=True), backend)
    simulator = Simulator(wrapper=wrapper, data=data.groupby("ppt"), parameters=draws)
    np.random.seed(seed)
    simulator.run()
    return data, simulator.export()


@pytest.mark.parametrize("backend", BACKENDS)
def test_rlrw_returns_the_choices_it_simulates(backend):
    """The sampled choices follow the policy, and the reward is that of the chosen arm."""
    data, simulated = simulate_rlrw(backend)
    response = simulated["response"].to_numpy()
    assert simulated["response"].dtype == np.int64
    ## the uniforms are drawn per participant in turn, one per trial, as `choose` uses them
    np.random.seed(0)
    uniforms = np.random.random_sample(len(simulated))
    cdf = simulated[["policy_0", "policy_1"]].cumsum(axis=1).to_numpy()
    np.testing.assert_array_equal(response, (cdf / cdf[:, -1:] <= uniforms[:, None]).sum(axis=1))
    rewards = data[["reward_left", "reward_right"]].to_numpy()
    np.testing.assert_array_equal(simulated["reward"], rewards[np.arange(len(data)), response])
    ## not all the same choice
    assert 0 < response.mean() < 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_rlrw_returns_the_observed_choices_when_not_generating(backend):
    data = bandit()
    model = on_backend(RLRW(data=data, dimensions=4, parameters_settings=[[0.3, 0, 1], [4, 0, 10]]), backend)
    model.run()
    np.testing.assert_array_equal(model.export()["response"], data["response"])
    assert model.simulation[0]["response"] == data["response"][0]


@pytest.mark.skipif(not _jit.JIT_ENABLED, reason="numba is not installed or disabled")
def test_rlrw_simulates_the_same_choices_with_and_without_numba():
    _, python = simulate_rlrw("python", seed=11)
    _, numba = simulate_rlrw("numba", seed=11)
    np.testing.assert_array_equal(python["response"], numba["response"])


@pytest.mark.skipif(not _jit.JIT_ENABLED, reason="numba is not installed or disabled")
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_backends_agree(case):
    exports = []
    for backend in ("python", "numba"):
        model, _ = build(case, backend)
        np.random.seed(7)
        model.run()
        exports.append(model.export())
    pd.testing.assert_frame_equal(exports[0], exports[1], check_exact=False, **TOLERANCE)


def test_numba_is_used_when_installed(monkeypatch):
    model = RLRW(data=bandit(), dimensions=4)
    assert model._session_model.backend == ("numba" if _jit.JIT_ENABLED else "python")
    monkeypatch.setattr(_jit, "HAVE_NUMBA", False)
    monkeypatch.setattr(_jit, "JIT_ENABLED", False)
    assert RLRW(data=bandit(), dimensions=4)._session_model.backend == "python"


def test_a_numba_that_fails_to_load_falls_back_to_python_with_a_warning(monkeypatch):
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "numba":
            raise OSError("llvmlite could not be loaded")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("CPM_DISABLE_JIT", raising=False)
    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.setattr(_jit, "_STATE", {})
    with pytest.warns(UserWarning, match="could not be imported"):
        assert _jit._probe()["HAVE_NUMBA"] is False


@pytest.mark.skipif(not _jit.JIT_ENABLED, reason="numba is not installed or disabled")
def test_kernels_compile_where_numba_cannot_cache(monkeypatch):
    from numba.core import caching

    monkeypatch.setattr(caching.CacheImpl, "_locator_classes", [])

    def double(x):
        return 2 * x

    assert _jit.njit(double)(2) == 4


def test_cpm_imports_from_a_zip(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[2]
    archive = tmp_path / "cpm.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in (root / "cpm").rglob("*.py"):
            z.write(path, path.relative_to(root))
    code = "import cpm, cpm.models; assert 'cpm.zip' in cpm.__file__"
    env = {**os.environ, "PYTHONPATH": str(archive)}
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)


def test_missing_columns_are_reported_when_the_model_runs():
    """As before, the data are checked when they are used, not when the model is created."""
    model = PTSM2025(data=risky().drop(columns="ambiguity"), parameters_settings=PT)
    with pytest.raises(KeyError, match="ambiguity"):
        model.run()
    model = HybridMBMF(data=two_step().drop(columns="reward_0"), generate=True)
    with pytest.raises(KeyError, match="reward_0"):
        model.run()


def _recoded(data, **columns):
    data = data.copy()
    for key, value in columns.items():
        data[key] = value(data[key])
    return data


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("make, match", [
    (lambda: RLRW(data=_recoded(bandit(), response=lambda x: x + 1), dimensions=4), "response"),
    (lambda: RLRW(data=_recoded(bandit(), response=lambda x: x.where(x.index != 3)), dimensions=4),
     "response"),
    (lambda: RLRW(data=bandit(), dimensions=2), "stimuli"),
    (lambda: HybridMBMF(data=_recoded(two_step(), s1=lambda x: x + 1)), "s1"),
    (lambda: HybridMBMF(data=_recoded(two_step(), action=lambda x: x + 1)), "action"),
    (lambda: PTSM(data=_recoded(risky(), observed=lambda x: x + 1), parameters_settings=PT),
     "observed"),
], ids=["RLRW-1-based", "RLRW-NaN", "RLRW-dimensions", "Hybrid-s1", "Hybrid-action", "PTSM"])
def test_data_out_of_range_raise_instead_of_reading_past_the_arrays(make, match, backend):
    """numba does not check indices; the models check the data they index with."""
    model = on_backend(make(), backend)
    with pytest.raises(IndexError, match=match):
        model.run()


@pytest.mark.parametrize("backend", BACKENDS)
def test_nan_rewards_raise_as_softmax_does(backend):
    data = bandit()
    data["reward_left"] = data.reward_left.where(data.index != 3)
    model = on_backend(RLRW(data=data, dimensions=4), backend)
    with pytest.raises(ValueError, match="NaN"):
        model.run()


def test_models_survive_deepcopy_and_pickling():
    model = HybridMBMF(data=two_step(), parameters_settings=HYBRID)
    model.run()
    clone = pickle.loads(pickle.dumps(copy.deepcopy(model)))
    clone.reset()
    clone.run()
    np.testing.assert_array_equal(clone.dependent, model.dependent)
    assert len(clone.simulation) == len(model.simulation)


def test_parallel_fit_matches_the_serial_fit():
    """With multiprocess on Windows (spawn), each worker loads the cached kernels."""
    data = pd.concat([bandit(p).assign(ppt=p) for p in (1, 2, 3, 4)], ignore_index=True)
    fits = []
    for parallel in (False, True):
        fit = FminBound(
            model=RLRW(data=bandit(1), dimensions=4, parameters_settings=[[0.3, 0, 1], [4, 0, 10]]),
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
