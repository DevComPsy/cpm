"""
The benchmark cases: every built-in application of cpm, on one participant.

Each case builds a model for a given backend and returns it together with the
observed data and the parameter values at which the objective is evaluated.
`backend=None` is the per-trial application (the original classes); the other
backends are the session versions. Cases whose session version does not exist
in the installed cpm are skipped, so the same file also benchmarks older
versions of cpm.
"""

import warnings

import numpy as np
import pandas as pd

import cpm
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


def _application(module, name, backend):
    """The per-trial class for backend None, else its session version (or None)."""
    module = getattr(cpm.applications, module)
    if backend is None:
        return getattr(module, name)
    session = getattr(module, name + "Session", None)
    if session is None:
        return None
    return lambda **kwargs: session(backend=backend, **kwargs)


def build(case, backend=None):
    """
    The model, observed data and parameter values of a benchmark case.

    Returns None if the installed cpm has no model for this backend.
    """
    warnings.simplefilter("ignore")
    if case == "RLRW":
        cls = _application("reinforcement_learning", "RLRW", backend)
        data = bandit()
        kwargs = dict(dimensions=4, parameters_settings=[[0.3, 0, 1], [4, 0, 10]])
    elif case == "HybridMBMF":
        cls = _application("reinforcement_learning", "HybridMBMF", backend)
        data = two_step()
        kwargs = dict(
            parameters_settings=[
                [2, 0, 5], [0.4, 0, 1], [0.6, 0, 1], [0.5, 0, 1], [0.3, -5, 5], [-0.2, -5, 5]
            ]
        )
    elif case in ("PTSM", "PTSM1992", "PTSM2025"):
        cls = _application("decision_making", case, backend)
        data = risky()
        kwargs = dict(parameters_settings=PT_SETTINGS)
    else:
        raise ValueError(case)
    if cls is None:
        return None
    model = cls(data=data, **kwargs)
    observed = data["observed"].to_numpy()
    x = np.array([model.parameters[k].value for k in model.parameters.free()], dtype=float)
    return model, observed, x


CASES = ["RLRW", "HybridMBMF", "PTSM", "PTSM1992", "PTSM2025"]
BACKENDS = [None, "python", "numba"]
