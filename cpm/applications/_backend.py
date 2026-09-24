"""
Shared machinery of the session versions of the built-in applications.
"""

import numpy as np
import pandas as pd

from ..core import _jit
from ..generators.session import session_data

__all__ = ["SessionModel", "require", "uniforms"]


class SessionModel:
    """
    A model function for `SessionWrapper` that runs a loop of `cpm.applications._sessions`.

    Subclasses implement `__call__(parameters, data)`. The loops are looked up
    by backend when first needed, rather than stored, so that the model function
    pickles by reference and can be sent to other processes for parallel fits.

    Parameters
    ----------
    backend : str
        "numba" or "python", as returned by `cpm.core._jit.resolve_backend`.
    """

    def __init__(self, backend):
        self.backend = backend
        self._sessions = None

    @property
    def sessions(self):
        if self._sessions is None:
            self._sessions = _jit.kernels(
                "cpm.applications._sessions", python=self.backend == "python"
            )
        return self._sessions

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sessions"] = None
        return state


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
