"""
Shared machinery of the built-in applications.

Each application computes all trials of a participant in one call of a loop in
`cpm.applications._sessions`, compiled with numba if numba is installed and
plain Python otherwise. The `Application` base class keeps the interface of a
per-trial `cpm.generators.Wrapper` on top of that:

- `model` is a per-trial model function, `model(parameters, trial)`, which
  returns the outputs of one trial as a dictionary. It runs the same loop on that
  single trial, so there is only one implementation of each model.
- `simulation` is the list of these per-trial dictionaries for the last run,
  built from the outputs of the run when it is first read.
"""

import copy

import numpy as np
import pandas as pd

from ..core import _jit
from ..generators.session import SessionWrapper, session_data, trial_view

__all__ = ["Application", "SessionModel", "TrialModel", "indices", "require", "uniforms"]


class SessionModel:
    """
    A model function for `SessionWrapper` that runs a loop of `cpm.applications._sessions`.

    Subclasses implement `__call__(parameters, data)`. The loops are looked up
    by backend when needed, rather than stored, so that the model function pickles
    by reference and can be sent to other processes for parallel fits.

    Parameters
    ----------
    generate : bool
        Whether the model samples its choices.

    Attributes
    ----------
    backend : str
        "numba" if numba is installed and not disabled, "python" otherwise.
    """

    def __init__(self, generate=False):
        self.generate = generate
        self.backend = _jit.resolve_backend("auto")

    @property
    def sessions(self):
        return _jit.kernels("cpm.applications._sessions", python=self.backend == "python")


def one_trial(trial):
    """The data of a single trial (a `pandas.Series` or a dict) as one-row data."""
    items = trial.items() if isinstance(trial, (pd.Series, dict)) else dict(trial).items()
    return {key: np.asarray([value]) for key, value in items}


class TrialModel:
    """
    The per-trial model function of an application: its session model, run on one trial.

    Called as `model(parameters, trial)`, like the model function of a per-trial
    `Wrapper`, it computes one trial from the states in `parameters` and returns
    that trial's outputs, including the updated states.
    """

    def __init__(self, session_model, prepare, shapes):
        self.session_model = session_model
        self.prepare = prepare
        self.shapes = shapes

    def __call__(self, parameters, trial, generate=None):
        model, prepare = self.session_model, self.prepare
        if generate is not None and generate != model.generate:
            model = copy.copy(model)
            model.generate = generate
            if hasattr(prepare, "generate"):
                prepare = copy.copy(prepare)
                prepare.generate = generate
        output = model(parameters=parameters, data=prepare(one_trial(trial)))
        return trial_view(output, 0, self.shapes)


class Application(SessionWrapper):
    """
    The base class of the built-in applications, see the module docstring.

    Subclasses call `_setup` from their constructor, and set `_trial_shapes` for
    outputs whose per-trial value is not simply a row of the session output.
    """

    def _setup(self, data, parameters, session_model, prepare):
        super().__init__(model=session_model, data=data, parameters=parameters, prepare=prepare)
        self._session_model = session_model
        self.model = TrialModel(session_model, prepare, self._trial_shapes)

    def _run_session(self):
        return self._session_model(parameters=self.parameters, data=self.session)


def require(data, columns, model):
    """Raise a helpful error if the prepared data lack any of `columns`."""
    missing = [column for column in columns if column not in data]
    if missing:
        raise KeyError(
            f"{model} needs the column(s) {missing} in the data, which has "
            f"{sorted(data)}."
        )


def uniforms(trials, needed):
    """One uniform random number per trial from `numpy.random` if `needed`, else an empty array."""
    if needed:
        return np.random.random_sample(trials)
    return np.empty(0)


def indices(values):
    """
    `values` as int64 indices.

    Casting NaN or infinity to an integer gives a different number on different
    platforms (0 on ARM), so they become an index out of any range instead, which
    the loops of `cpm.applications._sessions` reject where they use it.
    """
    values = np.asarray(values)
    if values.dtype.kind == "f":
        values = np.where(np.isfinite(values), values, np.iinfo(np.int32).min)
    return np.ascontiguousarray(values, dtype=np.int64)


def prepared(data):
    """`session_data`, with integer and boolean columns as int64 and all others as float64 where possible."""
    out = {}
    for key, value in session_data(data).items():
        if value.dtype.kind in "iub":
            out[key] = np.ascontiguousarray(value, dtype=np.int64)
        elif value.dtype.kind == "f":
            out[key] = np.ascontiguousarray(value, dtype=np.float64)
        else:
            out[key] = value
    return out


def ordered_columns(data, pattern):
    """The columns of `data` whose name contains `pattern`, in the order of the data."""
    if isinstance(data, pd.DataFrame):
        names = list(data.columns)
    else:
        names = [k for k in data if k != "ppt"]
    return [name for name in names if pattern in name]
