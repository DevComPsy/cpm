"""
The benchmark cases: every built-in application of cpm, on one participant.

Each case builds a model for a given backend and returns it together with the
observed data and the parameter values at which the objective is evaluated.

- "python" and "numba": the application, which computes all trials at once, as
  plain Python or compiled with numba. Installing numba makes "numba" the default.
- "trial": the application's per-trial `model` function in a plain per-trial
  `Wrapper`, which is how users extend an application trial by trial.

With cpm versions before the applications computed all trials at once, only
"trial" is available, and it is the application itself.
"""

import warnings

import numpy as np
import pandas as pd

import cpm
from cpm.generators import Wrapper
from cpm.datasets import load_bandit_data, load_risky_choices


def bandit():
    data = load_bandit_data()
    data = data[data.ppt == 1].reset_index(drop=True)
    data["observed"] = data["response"]
    return data


def two_step(n=200, seed=0):
    rng = np.random.default_rng(seed)
    action = rng.integers(0, 2, n)
    common = rng.random(n) < 1.0  # deterministic transitions in the novel task
    return pd.DataFrame(
        {
            "s1": rng.integers(0, 2, n),
            "stimuli_first": rng.integers(0, 2, n),
            "action": action,
            "s2": np.where(common, 1 - action, action),
            "reward": rng.integers(0, 10, n) / 9,
            "reward_0": rng.integers(0, 10, n) / 9,
            "reward_1": rng.integers(0, 10, n) / 9,
            "observed": action,
        }
    )


def risky():
    data = load_risky_choices()
    data = data[data.ppt == 1].reset_index(drop=True)
    data["observed"] = data["choice"].astype(int)
    return data


PT_SETTINGS = {
    "alpha": [0.8, 1e-2, 5.0],
    "lambda_loss": [1.6, 1e-2, 5.0],
    "gamma": [0.7, 1e-2, 5.0],
    "temperature": [3.0, 1e-2, 15.0],
    "beta": [0.9, 0.0, 5.0],
    "delta": [0.6, 1e-2, 5.0],
    "eta": [0.1, -0.49, 0.49],
    "phi_gain": [0.2, -10.0, 10.0],
    "phi_loss": [-0.3, -10.0, 10.0],
}


def _application(module, name):
    return getattr(getattr(cpm.applications, module), name)


def _on_backend(model, backend):
    """The model on `backend`, or None if this version of cpm cannot run it there."""
    session = getattr(model, "_session_model", None)
    if backend == "trial":
        if session is None:
            return model  # an older cpm: the application is a per-trial Wrapper
        return Wrapper(model=model.model, data=model.data, parameters=model.parameters)
    if session is None:
        return None
    session.backend = backend
    return model


def build(case, backend="trial"):
    """
    The model, observed data and parameter values of a benchmark case.

    Returns None if the installed cpm has no model for this backend.
    """
    warnings.simplefilter("ignore")
    if case == "RLRW":
        cls = _application("reinforcement_learning", "RLRW")
        data = bandit()
        kwargs = dict(dimensions=4, parameters_settings=[[0.3, 0, 1], [4, 0, 10]])
    elif case == "HybridMBMF":
        cls = _application("reinforcement_learning", "HybridMBMF")
        data = two_step()
        kwargs = dict(
            parameters_settings=[
                [2, 0, 5], [0.4, 0, 1], [0.6, 0, 1], [0.5, 0, 1], [0.3, -5, 5], [-0.2, -5, 5]
            ]
        )
    elif case in ("PTSM", "PTSM1992", "PTSM2025"):
        cls = _application("decision_making", case)
        data = risky()
        kwargs = dict(parameters_settings=PT_SETTINGS)
    else:
        raise ValueError(case)
    if backend == "numba" and not _jit_enabled():
        return None
    model = _on_backend(cls(data=data, **kwargs), backend)
    if model is None:
        return None
    observed = data["observed"].to_numpy()
    x = np.array([model.parameters[k].value for k in model.parameters.free()], dtype=float)
    return model, observed, x


def _jit_enabled():
    try:
        from cpm.core import _jit
    except ImportError:
        return False
    return _jit.JIT_ENABLED


CASES = ["RLRW", "HybridMBMF", "PTSM", "PTSM1992", "PTSM2025"]
BACKENDS = ["trial", "python", "numba"]
